import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.longformer.modeling_longformer import (
    LongformerPreTrainedModel,
    LongformerLMHead,
)

logger = logging.getLogger(__name__)


class RecformerConfig(RobertaConfig):
    def __init__(
        self,
        *_,
        temp: float | None = None,
        max_attr_num: int | None = None,
        max_attr_length: int | None = None,
        max_item_embeddings: int | None = None,  # Maximum number of items
        max_token_num: int | None = None,  # Maximum number of tokens
        item_num: int | None = None,  # Number of items in the dataset
        finetune_negative_sample_size: int | None = None,  # Number of negative samples for finetuning
        pooler_type: str | None = None,  # Pooler type
        session_reduce_method: str | None = None,
        session_reduce_topk: int | None = None,
        session_reduce_weightedsim_temp: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.temp = temp
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length
        self.max_item_embeddings = max_item_embeddings
        self.max_token_num = max_token_num
        self.item_num = item_num
        self.finetune_negative_sample_size = finetune_negative_sample_size
        self.pooler_type = pooler_type
        self.session_reduce_method = session_reduce_method
        self.session_reduce_topk = session_reduce_topk
        self.session_reduce_weightedsim_temp = session_reduce_weightedsim_temp


@dataclass
class RecformerPretrainingOutput:

    cl_correct_num: float = 0.0
    cl_total_num: float = 1e-5
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class RecformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: RecformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.token_type_size, config.hidden_size)
        self.item_position_embeddings = nn.Embedding(config.max_item_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.original_embedding = config.original_embedding

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, item_position_ids=None, inputs_embeds=None
    ):
        def original_forward():
            nonlocal position_ids, token_type_ids, input_ids, inputs_embeds

            if position_ids is None:
                if input_ids is not None:
                    # Create the position ids from the input token ids. Any padded tokens remain padded.
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + position_embeddings + token_type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

        def recformer_forward():
            nonlocal position_ids, token_type_ids, input_ids, inputs_embeds

            if position_ids is None:
                if input_ids is not None:
                    # Create the position ids from the input token ids. Any padded tokens remain padded.
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            item_position_embeddings = self.item_position_embeddings(item_position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + item_position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

        if self.original_embedding:
            return original_forward()
        else:
            return recformer_forward()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor inputs_embeds:
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RecformerPooler(nn.Module):
    def __init__(self, config: RecformerConfig):
        super().__init__()
        assert config.pooler_type in ["cls", "token", "item", "attribute"]

        self.pooler_type = config.pooler_type
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        attr_type_ids: torch.Tensor,
        item_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooler_type == "attribute":
            # attention_mask: (bs, seq_len)
            # hidden_states: (bs, seq_len, hidden_size)
            # attr_type_ids: (bs, seq_len)
            attr_max = attr_type_ids.max()

            num_items = item_position_ids.clone()
            num_items[num_items == 50] = -100
            num_items = torch.max(num_items, dim=1).values  # (bs, )
            items_max = num_items.max()

            # Create boolean masks for attribute and item
            # attr_mask  (bs, attr_num, seq_len)
            # item_mask  (bs, item_num, seq_len)
            # attr_item_mask  (bs, attr_num, items_max, seq_len)
            attr_mask = torch.eq(
                attr_type_ids.unsqueeze(1), torch.arange(1, attr_max + 1, device=attr_type_ids.device).reshape(1, -1, 1)
            )  # (bs, attr_num, seq_len)
            item_mask = torch.eq(
                item_position_ids.unsqueeze(1),
                torch.arange(1, items_max + 1, device=item_position_ids.device).reshape(1, -1, 1),
            )  # (bs, item_num, seq_len)
            attr_item_mask = torch.mul(attr_mask.unsqueeze(2), item_mask.unsqueeze(1))

            # Select hidden_state for each item and each attribute
            # Vectorized implementation
            # attr_item_mask  (bs, attr_num, items_max, seq_len)  Boolean mask
            # hidden_states  (bs, seq_len, hidden_size)
            # hidden_states_pooled  (bs, attr_num, items_max, hidden_size)
            hidden_states_pooled = hidden_states.unsqueeze(1).unsqueeze(2) * attr_item_mask.unsqueeze(-1)
            hidden_states_pooled[~attr_item_mask] = torch.nan
            # (bs, attr_num, items_max, seq_len, hidden_size)
            hidden_states_pooled = hidden_states_pooled.nanmean(dim=3)  # (bs, attr_num, items_max, hidden_size)

        elif self.pooler_type == "item":
            num_items = item_position_ids.clone()
            num_items[num_items == 50] = -100
            num_items = torch.max(num_items, dim=1).values  # (bs, )
            items_max = num_items.max()

            # item_mask  (bs, item_num, seq_len)
            item_mask = torch.eq(
                item_position_ids.unsqueeze(1),
                torch.arange(1, items_max + 1, device=item_position_ids.device).reshape(1, -1, 1),
            )  # (bs, item_num, seq_len)

            # hidden_states  (bs, seq_len, hidden_size)
            # hidden_states_pooled  (bs, items_max, hidden_size)
            hidden_states_pooled = torch.mul(
                hidden_states.unsqueeze(1), item_mask.unsqueeze(-1)
            )  # (bs, item_num, seq_len, hidden_size)
            hidden_states_pooled[~item_mask] = torch.nan  # (bs, items_max, seq_len, hidden_size)
            hidden_states_pooled = hidden_states_pooled.nanmean(dim=2)  # (bs, items_max, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1)  # (bs, 1, items_max, hidden_size)

        elif self.pooler_type == "token":
            seq_len = hidden_states.shape[1]

            mask = attention_mask[..., :seq_len].bool()  # (bs, seq_len)
            hidden_states_pooled = hidden_states  # (bs, seq_len, hidden_size)
            hidden_states_pooled[~mask] = torch.nan  # (bs, seq_len, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1)  # (bs, 1, seq_len, hidden_size)

        elif self.pooler_type == "cls":
            hidden_states_pooled = hidden_states[:, 0, :]  # (bs, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, hidden_size)

        else:
            raise ValueError(f"pooler_type {self.pooler_type} is not supported")

        return hidden_states_pooled


class RecformerModelWithPooler(nn.Module):
    def __init__(self, config: RecformerConfig):
        super().__init__()
        self.config = config

        self.model = RobertaModel(config, add_pooling_layer=False)
        self.pooler = RecformerPooler(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        attr_type_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        model_output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = model_output.last_hidden_state

        pooled_output = self.pooler(
            attention_mask=attention_mask,
            hidden_states=sequence_output,
            attr_type_ids=attr_type_ids,
            item_position_ids=item_position_ids,
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config: RecformerConfig):
        super().__init__()
        self.temp = config.temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class RecformerForPretraining(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)

        self.longformer = RecformerModelWithPooler(config)
        self.lm_head = LongformerLMHead(config)
        self.sim = Similarity(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids_a: Optional[torch.Tensor] = None,
        attention_mask_a: Optional[torch.Tensor] = None,
        global_attention_mask_a: Optional[torch.Tensor] = None,
        token_type_ids_a: Optional[torch.Tensor] = None,
        attr_type_ids_a: Optional[torch.Tensor] = None,
        item_position_ids_a: Optional[torch.Tensor] = None,
        mlm_input_ids_a: Optional[torch.Tensor] = None,
        mlm_labels_a: Optional[torch.Tensor] = None,
        input_ids_b: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        global_attention_mask_b: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
        attr_type_ids_b: Optional[torch.Tensor] = None,
        item_position_ids_b: Optional[torch.Tensor] = None,
        mlm_input_ids_b: Optional[torch.Tensor] = None,
        mlm_labels_b: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids_a.size(0)

        outputs_a = self.longformer(
            input_ids_a,
            attention_mask=attention_mask_a,
            global_attention_mask=global_attention_mask_a,
            head_mask=head_mask,
            token_type_ids=token_type_ids_a,
            attr_type_ids=attr_type_ids_a,
            position_ids=position_ids,
            item_position_ids=item_position_ids_a,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        outputs_b = self.longformer(
            input_ids_b,
            attention_mask=attention_mask_b,
            global_attention_mask=global_attention_mask_b,
            head_mask=head_mask,
            token_type_ids=token_type_ids_b,
            attr_type_ids=attr_type_ids_b,
            position_ids=position_ids,
            item_position_ids=item_position_ids_b,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # MLM auxiliary objective
        mlm_outputs_a = None
        if mlm_input_ids_a is not None:
            mlm_outputs_a = self.longformer(
                mlm_input_ids_a,
                attention_mask=attention_mask_a,
                global_attention_mask=global_attention_mask_a,
                head_mask=head_mask,
                token_type_ids=token_type_ids_a,
                attr_type_ids=attr_type_ids_a,
                position_ids=position_ids,
                item_position_ids=item_position_ids_a,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        mlm_outputs_b = None
        if mlm_input_ids_b is not None:
            mlm_outputs_b = self.longformer(
                mlm_input_ids_b,
                attention_mask=attention_mask_b,
                global_attention_mask=global_attention_mask_b,
                head_mask=head_mask,
                token_type_ids=token_type_ids_b,
                attr_type_ids=attr_type_ids_b,
                position_ids=position_ids,
                item_position_ids=item_position_ids_b,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        z1 = outputs_a.pooler_output  # (bs*num_sent, hidden_size)
        z2 = outputs_b.pooler_output  # (bs*num_sent, hidden_size)

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        correct_num = (torch.argmax(cos_sim, 1) == labels).sum()

        if mlm_outputs_a is not None and mlm_labels_a is not None:
            mlm_labels_a = mlm_labels_a.view(-1, mlm_labels_a.size(-1))
            prediction_scores_a = self.lm_head(mlm_outputs_a.last_hidden_state)
            masked_lm_loss_a = loss_fct(prediction_scores_a.view(-1, self.config.vocab_size), mlm_labels_a.view(-1))
            loss = loss + self.config.mlm_weight * masked_lm_loss_a

        if mlm_outputs_b is not None and mlm_labels_b is not None:
            mlm_labels_b = mlm_labels_b.view(-1, mlm_labels_b.size(-1))
            prediction_scores_b = self.lm_head(mlm_outputs_b.last_hidden_state)
            masked_lm_loss_b = loss_fct(prediction_scores_b.view(-1, self.config.vocab_size), mlm_labels_b.view(-1))
            loss = loss + self.config.mlm_weight * masked_lm_loss_b

        return RecformerPretrainingOutput(
            loss=loss,
            logits=cos_sim,
            cl_correct_num=correct_num,
            cl_total_num=batch_size,
            hidden_states=outputs_a.hidden_states,
            attentions=outputs_a.attentions,
            global_attentions=outputs_a.global_attentions,
        )


class RecformerForSeqRec(nn.Module):
    def __init__(self, config: RecformerConfig):
        super().__init__()
        self.config = config
        self.model_with_pooler = RecformerModelWithPooler(config)
        self.sim = Similarity(config)

        self.item_embedding = None

    def init_item_embedding(self, embeddings: Optional[torch.Tensor] = None):
        if embeddings is None:
            raise ValueError("embeddings must be provided.")

        self.item_embedding = embeddings.clone()
        self.item_embedding.requires_grad = False

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is not None:
            raise NotImplementedError("Negative sampling disabled")

        candidate_embeddings = self.item_embedding
        pooler_output = pooler_output.unsqueeze(1)  # (batch_size, 1, attr_num, items_max, hidden_size)
        sim = self.sim(pooler_output, candidate_embeddings)  # (batch_size, |I|, attr_num, items_max)

        return sim

    def forward_batch(self, batch, session_len, labels):
        attr_type_ids = batch["attr_type_ids"]
        token_type_ids = batch["token_type_ids"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        model_output = self.model_with_pooler.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # (bs, seq_len, hidden_size)

        hidden_states = model_output.last_hidden_state  # (bs, seq_len, hidden_size)

        session_index_start = (torch.cumsum(session_len, dim=0) - session_len).tolist()
        session_index_end = torch.cumsum(session_len, dim=0).tolist()

        num_session = len(session_len)
        max_session_len = max(session_len)
        max_seq_len, hidden_size = hidden_states.shape[1:]

        unfolded_hidden_states = torch.empty(
            (num_session, max_session_len * max_seq_len, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        unfolded_token_type_ids = torch.zeros(
            (num_session, max_session_len * max_seq_len),
            device=token_type_ids.device,
            dtype=token_type_ids.dtype,
        )
        unfolded_attr_type_ids = torch.zeros(
            (num_session, max_session_len * max_seq_len),
            device=attr_type_ids.device,
            dtype=attr_type_ids.dtype,
        )
        unfolded_attention_mask = torch.zeros(
            (num_session, max_session_len * max_seq_len),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        item_position_ids = (
            (torch.arange(max_session_len) + 1).unsqueeze(1).repeat(1, max_seq_len).reshape(-1).repeat(num_session, 1)
        )

        for session_idx, (start, end) in enumerate(zip(session_index_start, session_index_end)):
            hidden_state = hidden_states[start:end]  # (session_len, seq_len, hidden_size)
            hidden_state = hidden_state.reshape(-1, hidden_size)  # (session_len*seq_len, hidden_size)
            unfolded_hidden_states[session_idx, : hidden_state.shape[0]] = hidden_state

            token_type_id = token_type_ids[start:end]  # (session_len, seq_len)
            token_type_id = token_type_id.reshape(-1)  # (session_len*seq_len)
            unfolded_token_type_ids[session_idx, : token_type_id.shape[0]] = token_type_id

            attr_type_id = attr_type_ids[start:end]  # (session_len, seq_len)
            attr_type_id = attr_type_id.reshape(-1)  # (session_len*seq_len)
            unfolded_attr_type_ids[session_idx, : attr_type_id.shape[0]] = attr_type_id

            attention_mask_ = attention_mask[start:end]  # (session_len, seq_len)
            attention_mask_ = attention_mask_.reshape(-1)  # (session_len*seq_len)
            unfolded_attention_mask[session_idx, : attention_mask_.shape[0]] = attention_mask_

        item_position_ids[unfolded_token_type_ids == 0] = 50

        pooler_output = self.model_with_pooler.pooler.forward(
            attention_mask=unfolded_attention_mask,
            hidden_states=unfolded_hidden_states,
            attr_type_ids=unfolded_attr_type_ids,
            item_position_ids=item_position_ids.to(unfolded_hidden_states.device),
        )

        if labels is None:
            return self.similarity_score(pooler_output, candidates=None)

        return self._calculate_loss(pooler_output, labels=labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attr_type_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,  # candidate item ids
        labels: Optional[torch.Tensor] = None,  # target item ids
    ):
        outputs = self.model_with_pooler.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            attr_type_ids=attr_type_ids,
            item_position_ids=item_position_ids,
        )

        pooler_output = outputs.pooler_output  # (bs, hidden_size)

        return self._calculate_loss(pooler_output, labels)

    def _calculate_loss(self, pooler_output, labels):
        loss_fct = CrossEntropyLoss()

        if self.config.finetune_negative_sample_size <= 0:  ## using full softmax
            scores = self.similarity_score(pooler_output)  # (bs, |I|, attr_num, items_max)
            scores = reduce_session(scores, self.config.session_reduce_method, self.config.session_reduce_topk)

            if labels.dim() == 2:
                labels = labels.squeeze(dim=-1)
            loss = loss_fct(scores, labels)

        else:  ## using sampled softmax
            raise NotImplementedError("Negative sampling disabled")
            # candidates = torch.cat(
            #     (
            #         labels.unsqueeze(-1),
            #         torch.randint(
            #             0, self.config.item_num, size=(batch_size, self.config.finetune_negative_sample_size)
            #         ).to(labels.device),
            #     ),
            #     dim=-1,
            # )
            # scores = self.similarity_score(pooler_output, candidates)
            # target = torch.zeros_like(labels, device=labels.device)
            # loss = loss_fct(scores, target)

        return loss


def reduce_session(
    scores: torch.Tensor,
    session_reduce_method: str,
    session_reduce_topk: int | None = None,
    session_reduce_weightedsim_temp: float = 1.0,
):
    scores: torch.Tensor  # (bs, |I|, attr_num, items_max)

    if session_reduce_method == "maxsim":
        # Replace NaN with -inf
        scores[torch.isnan(scores)] = -torch.inf  # (bs, |I|, attr_num, items_max)
        scores = scores.max(dim=-1).values  # (bs, |I|, num_attr)
    elif session_reduce_method == "mean":
        scores = scores.nanmean(dim=-1)  # (bs, |I|, num_attr)
    elif session_reduce_method == "weightedsim":
        scores[torch.isnan(scores)] = -torch.inf
        scores = scores / session_reduce_weightedsim_temp  # (bs, |I|, num_attr, items_max)
        weights = torch.softmax(scores, dim=-1)  # (bs, |I|, num_attr, items_max)
        scores = torch.mul(scores, weights)  # (bs, |I|, num_attr, items_max)
        scores = scores.nanmean(dim=-1)  # (bs, |I|, num_attr)
    elif session_reduce_method == "topksim":
        session_reduce_topk = min(session_reduce_topk, scores.shape[-1])
        scores[torch.isnan(scores)] = -torch.inf
        scores = scores.topk(session_reduce_topk, dim=-1).values  # (bs, |I|, num_attr, topk)
        # Mask out any -inf left
        scores[scores == -torch.inf] = torch.nan
        scores = scores.nanmean(dim=-1)  # (bs, |I|, num_attr)
    else:
        raise ValueError("Unknown session reduce method.")

    scores = scores.mean(dim=-1)  # (bs, |I|)
    return scores
