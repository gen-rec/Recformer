import logging
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple

import torch
import torch.nn as nn
from torch import distributed
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers.models.longformer.modeling_longformer import (
    LongformerConfig,
    LongformerPreTrainedModel,
    LongformerEncoder,
    LongformerBaseModelOutputWithPooling,
    LongformerLMHead,
)

logger = logging.getLogger(__name__)


class RecformerConfig(LongformerConfig):
    def __init__(
        self,
        attention_window: Union[List[int], int] = 64,
        sep_token_id: int = 2,
        token_type_size: int = 4,  # <s>, key, value, <pad>
        max_token_num: int = 2048,
        max_item_embeddings: int = 32,  # 1 for <s>, 50 for items
        max_attr_num: int = 12,
        max_attr_length: int = 8,
        pooler_type: str = "cls",
        temp: float = 0.05,
        mlm_weight: float = 0.1,
        item_num: int = 0,
        finetune_negative_sample_size: int = 0,
        **kwargs,
    ):
        super().__init__(attention_window, sep_token_id, **kwargs)

        self.token_type_size = token_type_size
        self.max_token_num = max_token_num
        self.max_item_embeddings = max_item_embeddings
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length
        self.pooler_type = pooler_type
        self.temp = temp
        self.mlm_weight = mlm_weight

        # finetune config

        self.item_num = item_num
        self.finetune_negative_sample_size = finetune_negative_sample_size

@dataclass
class RecformerPretrainingOutput:

    cl_correct_num: float = 0.0
    cl_total_num: float = 1e-5
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RecformerBaseModelOutputWithPooling(LongformerBaseModelOutputWithPooling):
    mask: Optional[torch.Tensor] = None
    linear_hidden_states: Optional[torch.Tensor] = None


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

        try:
            self.original_embedding = config.original_embedding
        except AttributeError:
            self.original_embedding = None

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if self.original_embedding:
            self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        else:
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

        try:
            self.original_embedding = config.original_embedding
        except AttributeError:
            self.original_embedding = None

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

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)

            embeddings = inputs_embeds + position_embeddings
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

        self.linear = nn.Linear(config.hidden_size, config.linear_out)

    def forward(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        attr_type_ids: torch.Tensor,
        item_position_ids: torch.Tensor,
    ):
        hidden_states = self.linear.forward(hidden_states)

        if self.pooler_type == "attribute":
            attr_max = attr_type_ids.max()
            num_items = item_position_ids.clone()
            num_items[num_items == 50] = -100
            num_items = torch.max(num_items, dim=1).values
            items_max = num_items.max()

            attr_mask = torch.eq(
                attr_type_ids.unsqueeze(1), torch.arange(1, attr_max + 1, device=attr_type_ids.device).reshape(1, -1, 1)
            )  # (bs, attr_num, seq_len)
            item_mask = torch.eq(
                item_position_ids.unsqueeze(1),
                torch.arange(1, items_max + 1, device=item_position_ids.device).reshape(1, -1, 1),
            )
            attr_item_mask = torch.mul(attr_mask.unsqueeze(2), item_mask.unsqueeze(1))  # Ignore tokens that are False

            hidden_states_pooled = hidden_states.unsqueeze(1).unsqueeze(2) * attr_item_mask.unsqueeze(
                -1
            )  # (bs, attr_num, items_max, seq_len, hidden_size)

            # Sum across the required dimension
            summed_states = torch.sum(hidden_states_pooled, dim=3)  # Sum across the sequence length dimension

            # Count the number of valid (not masked out) elements in the attr_item_mask for each position
            valid_counts = attr_item_mask.sum(dim=3)
            valid_counts.unsqueeze_(-1)  # Adding an extra dimension to match the dimensionality for division

            valid_counts_eq_0 = torch.eq(valid_counts, 0)  # (bs, attr_num, items_max, 1)

            # Avoid division by zero by replacing 0 counts with 1
            valid_counts = torch.where(valid_counts_eq_0, torch.ones_like(valid_counts), valid_counts)

            # Compute the mean
            hidden_states_pooled = summed_states / valid_counts  # (bs, attr_num, items_max, hidden_size)

            return hidden_states_pooled, valid_counts_eq_0.squeeze(-1), hidden_states

        elif self.pooler_type == "item":
            num_items = item_position_ids.clone()
            num_items[num_items == 50] = -100
            num_items = torch.max(num_items, dim=1).values  # (bs, )
            items_max = num_items.max()

            item_mask = torch.eq(
                item_position_ids.unsqueeze(1),
                torch.arange(1, items_max + 1, device=item_position_ids.device).reshape(1, -1, 1),
            )  # (bs, item_num, seq_len)

            hidden_states_pooled = hidden_states.unsqueeze(1) * item_mask.unsqueeze(
                -1
            )  # (bs, item_num, seq_len, hidden_size)

            # Sum across the required dimension
            summed_states = torch.sum(hidden_states_pooled, dim=2)  # Sum across the sequence length dimension

            # Instead of using NaNs, use 0s and then use the mask to compute the mean
            valid_counts = item_mask.sum(dim=2).unsqueeze(
                -1
            )  # Count of True values for the mean along seq_len dimension

            valid_counts_eq_0 = torch.eq(valid_counts, 0)  # (bs, item_num, 1)

            # Avoid division by zero by replacing 0 counts with 1
            valid_counts = torch.where(valid_counts_eq_0, torch.ones_like(valid_counts), valid_counts)

            # Compute the mean
            hidden_states_pooled = summed_states / valid_counts  # (bs, item_num, hidden_size)

            return hidden_states_pooled.unsqueeze(1), valid_counts_eq_0.squeeze(-1).unsqueeze(1), hidden_states

        elif self.pooler_type == "token":
            seq_len = hidden_states.shape[1]

            mask = attention_mask[..., :seq_len].bool()  # (bs, seq_len)
            hidden_states_pooled = hidden_states  # (bs, seq_len, hidden_size)
            hidden_states_pooled[~mask] = torch.nan  # (bs, seq_len, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1)  # (bs, 1, seq_len, hidden_size)

            return hidden_states_pooled, mask, hidden_states

        elif self.pooler_type == "cls":
            hidden_states_pooled = hidden_states[:, 0, :]  # (bs, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, hidden_size)

            return hidden_states_pooled, torch.zeros(
                hidden_states_pooled.shape[:-1], dtype=torch.bool, device=hidden_states_pooled.device
            ), hidden_states

        else:
            raise ValueError(f"pooler_type {self.pooler_type} is not supported")


class RecformerModel(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = RecformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = RecformerPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        item_position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            # logger.info(
            #     f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
            #     f"`config.attention_window`: {attention_window}"
            # )
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if item_position_ids is None:
                unpadded_item_position_ids = None
            else:
                unpadded_item_position_ids = item_position_ids
                item_position_ids = nn.functional.pad(item_position_ids, (0, padding_len), value=pad_token_id)

            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0
        else:
            unpadded_item_position_ids = item_position_ids

        return (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            item_position_ids,
            inputs_embeds,
            unpadded_item_position_ids,
        )

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attr_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RecformerBaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            item_position_ids,
            inputs_embeds,
            unpadded_item_position_ids,
        ) = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[
            :, 0, 0, :
        ]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output, mask, linear_hidden_states = (
            self.pooler.forward(
                attention_mask=attention_mask,
                hidden_states=sequence_output,
                attr_type_ids=attr_type_ids,
                item_position_ids=unpadded_item_position_ids,
            )
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return RecformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            linear_hidden_states=linear_hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
            mask=mask,
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


class RecformerForPretraining(nn.Module):
    def __init__(self, config: RecformerConfig):
        super().__init__()

        self.config = config
        self.longformer = RecformerModel(config)
        self.lm_head = LongformerLMHead(config)
        self.sim = Similarity(config)

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        batch_size = input_ids_a.size(0)

        outputs_a = self.longformer.forward(
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
        outputs_b = self.longformer.forward(
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
            mlm_outputs_a = self.longformer.forward(
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
            mlm_outputs_b = self.longformer.forward(
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

        z1 = outputs_a.pooler_output  # (bs, attr_num, session_max, hidden_size)
        z2 = outputs_b.pooler_output  # (bs, attr_num, 1, hidden_size)

        z1_mask = outputs_a.mask  # (bs, attr_num, session_max)
        z2_mask = outputs_b.mask  # (bs, attr_num, 1)

        # Gather all embeddings if using distributed training
        if self.training and distributed.is_initialized():
            # Pad z1 to (bs, attr_num, max_item_embeddings, hidden_size)
            z1_padded = torch.empty(
                (batch_size, z1.size(1), self.config.max_item_embeddings, z1.size(3)), device=z1.device
            )
            z1_padded[:, :, : z1.size(2), :] = z1

            # Dummy vectors for allgather
            z1_list = [torch.empty_like(z1_padded) for _ in range(distributed.get_world_size())]
            z2_list = [torch.empty_like(z2) for _ in range(distributed.get_world_size())]

            # Allgather
            distributed.all_gather(tensor_list=z1_list, tensor=z1_padded.contiguous())
            distributed.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[distributed.get_rank()] = z1_padded
            z2_list[distributed.get_rank()] = z2

            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

            if z1_mask is not None and z2_mask is not None:
                z1_mask_padded = torch.ones(
                    (batch_size, z1_mask.size(1), self.config.max_item_embeddings),
                    dtype=torch.bool,
                    device=z1_mask.device,
                )
                z1_mask_padded[:, :, : z1_mask.size(2)] = z1_mask

                z1_mask_list = [torch.empty_like(z1_mask_padded) for _ in range(distributed.get_world_size())]
                z2_mask_list = [torch.empty_like(z2_mask) for _ in range(distributed.get_world_size())]

                distributed.all_gather(tensor_list=z1_mask_list, tensor=z1_mask_padded.contiguous())
                distributed.all_gather(tensor_list=z2_mask_list, tensor=z2_mask.contiguous())

                z1_mask = torch.cat(z1_mask_list, 0)
                z2_mask = torch.cat(z2_mask_list, 0)

        # z1: (bs, attr_num, items_max, hidden_size)
        # z2: (bs, attr_num, 1, hidden_size)
        # z1_mask: (bs, attr_num, items_max)
        # z2_mask: (bs, attr_num, 1)
        z1.unsqueeze_(1)  # (bs, 1, attr_num, items_max, hidden_size)
        z2.unsqueeze_(0)  # (1, bs, attr_num, 1, hidden_size)
        z1_mask.unsqueeze_(1)  # (bs, 1, attr_num, items_max)
        z2_mask.unsqueeze_(0)  # (1, bs, attr_num, 1)

        scores = self.sim.forward(z1, z2)  # (bs, bs, attr_num, items_max)

        mask = torch.add(z1_mask, z2_mask)  # (bs, bs, attr_num, items_max)
        scores[mask] = -torch.inf

        cos_sim = reduce_session(scores, self.config.session_reduce_method, self.config.session_reduce_topk)  # (bs, bs)
        labels = torch.arange(cos_sim.size(0), device=cos_sim.device)

        ce_loss = cross_entropy(cos_sim, labels)
        correct_num = torch.eq(torch.argmax(cos_sim, 1), labels).sum()

        if mlm_outputs_a is not None and mlm_labels_a is not None:
            mlm_labels_a = mlm_labels_a.view(-1, mlm_labels_a.size(-1))
            prediction_scores_a = self.lm_head(mlm_outputs_a.last_hidden_state)
            masked_lm_loss_a = cross_entropy(
                prediction_scores_a.view(-1, self.config.vocab_size), mlm_labels_a.view(-1)
            )
            mlm_loss_a = self.config.mlm_weight * masked_lm_loss_a
        else:
            mlm_loss_a = 0

        if mlm_outputs_b is not None and mlm_labels_b is not None:
            mlm_labels_b = mlm_labels_b.view(-1, mlm_labels_b.size(-1))
            prediction_scores_b = self.lm_head(mlm_outputs_b.last_hidden_state)
            masked_lm_loss_b = cross_entropy(
                prediction_scores_b.view(-1, self.config.vocab_size), mlm_labels_b.view(-1)
            )
            mlm_loss_b = self.config.mlm_weight * masked_lm_loss_b
        else:
            mlm_loss_b = 0

        loss = ce_loss + mlm_loss_a + mlm_loss_b

        return RecformerPretrainingOutput(
            loss=loss,
            logits=cos_sim,
            cl_correct_num=correct_num,
            cl_total_num=batch_size,
            hidden_states=outputs_a.hidden_states,
            attentions=outputs_a.attentions,
            global_attentions=outputs_a.global_attentions,
        )


class RecformerForSeqRec(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)

        self.longformer = RecformerModel(config)
        self.sim = Similarity(config)
        # Initialize weights and apply final processing
        self.item_embedding = None
        self.post_init()

        if config.attribute_agg_method == "linear":
            self.linear_agg_module = nn.Linear(3, 1)
        else:
            self.linear_agg_module = None

        self.gating_layer = nn.Sequential(
            nn.Linear(config.hidden_size, 3, bias=False), nn.Softmax(dim=-1)
        )

    def init_item_embedding(self, embeddings: Optional[torch.Tensor] = None):
        if embeddings is None:
            raise ValueError("embeddings must be provided.")

        self.item_embedding = nn.Parameter(embeddings, requires_grad=False)

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is not None:
            raise NotImplementedError("Negative sampling disabled")

        candidate_embeddings = self.item_embedding  # (|I|, attr_num, 1, hidden_size)
        pooler_output = pooler_output.unsqueeze(1)  # (batch_size, 1, attr_num, items_max, hidden_size)
        sim = self.sim(pooler_output, candidate_embeddings)  # (batch_size, |I|, attr_num, items_max)

        return sim

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attr_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        candidates: Optional[torch.Tensor] = None,  # candidate item ids
        labels: Optional[torch.Tensor] = None,  # target item ids
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)

        outputs = self.longformer.forward(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            attr_type_ids=attr_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooler_output = outputs.pooler_output  # (bs, attr_num, items_max, hidden_size)
        pooler_output_mask = outputs.mask  # (bs, attr_num, items_max)  True for valid tokens

        bos_embeddings = outputs.last_hidden_state[:, 0, :]  # (bs, hidden_size)
        weight = self.gating_layer(bos_embeddings)  # (bs, 4)

        if labels is None:
            scores = self.similarity_score(pooler_output)  # (bs, |I|, attr_num, items_max)

            all_item_mask = torch.zeros(
                (scores.shape[1], scores.shape[2], 1), dtype=torch.bool, device=scores.device
            )  # (|I|, attr_num, 1)
            final_mask = torch.add(pooler_output_mask.unsqueeze(1), all_item_mask.unsqueeze(0))
            scores[final_mask] = -torch.inf

            scores = reduce_session(
                scores,
                self.config.session_reduce_method,
                self.config.session_reduce_topk,
                mask=final_mask,
                attribute_agg_method="none",
                linear_agg_module=self.linear_agg_module,
            )

            scores = torch.mul(scores, weight.unsqueeze(1))  # (bs, |I|, attr_num + 1)
            scores = scores.sum(dim=2)  # (bs, |I|)

            return scores

        loss_fct = CrossEntropyLoss()

        if self.config.finetune_negative_sample_size <= 0:  ## using full softmax
            scores = self.similarity_score(pooler_output)  # (bs, |I|, attr_num, items_max)

            all_item_mask = torch.zeros((scores.shape[1], scores.shape[2], 1), dtype=torch.bool, device=scores.device)
            final_mask = torch.add(pooler_output_mask.unsqueeze(1), all_item_mask.unsqueeze(0))
            scores[final_mask] = -torch.inf

            scores = reduce_session(
                scores,
                self.config.session_reduce_method,
                self.config.session_reduce_topk,
                mask=final_mask,
                attribute_agg_method="none",
                linear_agg_module=self.linear_agg_module,
            )

            scores = torch.mul(scores, weight.unsqueeze(1))  # (bs, |I|, attr_num + 1)
            scores = scores.sum(dim=2)  # (bs, |I|)

            if labels.dim() == 2:
                labels = labels.squeeze(dim=-1)
            loss = loss_fct(scores, labels)

        else:  ## using sampled softmax
            raise NotImplementedError("Negative sampling disabled")
            candidates = torch.cat(
                (
                    labels.unsqueeze(-1),
                    torch.randint(
                        0, self.config.item_num, size=(batch_size, self.config.finetune_negative_sample_size)
                    ).to(labels.device),
                ),
                dim=-1,
            )
            scores = self.similarity_score(pooler_output, candidates)
            target = torch.zeros_like(labels, device=labels.device)
            loss = loss_fct(scores, target)

        return loss


def reduce_session(
    scores: torch.Tensor,
    session_reduce_method: str,
    session_reduce_topk: int | None = None,
    session_reduce_weightedsim_temp: float = 0.05,
    mask: torch.Tensor | None = None,
    attribute_agg_method: str = "mean",
    linear_agg_module: nn.Module | None = None,
):
    """
    Mask tensor: True to mask
    """
    scores: torch.Tensor  # (bs, |I|, attr_num, items_max)

    if session_reduce_method == "maxsim":
        # Replace NaN with -inf
        scores = scores.max(dim=-1).values  # (bs, |I|, num_attr)
    elif session_reduce_method == "mean":
        raise NotImplementedError("Mean pooling disabled")
        scores = scores.nanmean(dim=-1)  # (bs, |I|, num_attr)
    elif session_reduce_method == "weightedsim":
        raw_scores = scores
        scores = scores / session_reduce_weightedsim_temp  # (bs, |I|, num_attr, items_max)
        weights = torch.softmax(scores, dim=-1)  # (bs, |I|, num_attr, items_max)
        zero_weights_mask = torch.eq(weights, 0)  # (bs, |I|, num_attr, items_max)
        weights = torch.where(zero_weights_mask, torch.ones_like(weights), weights)  # Replace 0 with 1

        weighted_scores = torch.mul(scores, weights)  # (bs, |I|, num_attr, items_max)

        weighted_scores = torch.where(
            zero_weights_mask, torch.ones_like(weighted_scores), weighted_scores
        )  # Replace 0 with 1

        nonzero = torch.count_nonzero(weighted_scores, dim=-1)  # (bs, |I|, num_attr)
        weighted_scores_reduced = weighted_scores.sum(dim=-1) / nonzero

        if torch.any(torch.isnan(weighted_scores_reduced)):
            from IPython import embed

            embed()
            raise ValueError("NaN in scores")

        scores = weighted_scores_reduced

    elif session_reduce_method == "topksim":
        session_reduce_topk = min(session_reduce_topk, scores.shape[-1])
        scores = scores.topk(session_reduce_topk, dim=-1).values  # (bs, |I|, num_attr, topk)

        # Filter out any -inf
        inf_mask = torch.isinf(scores)  # (bs, |I|, num_attr, topk)
        # Replace inf with 0
        scores[inf_mask] = 0
        scores_nonzero = torch.count_nonzero(scores, dim=-1)  # (bs, |I|, num_attr)

        # Mean
        scores = scores.sum(dim=-1) / scores_nonzero  # (bs, |I|, num_attr)
    else:
        raise ValueError("Unknown session reduce method.")

    if attribute_agg_method == "mean":
        scores = scores.mean(dim=-1)
    elif attribute_agg_method == "max":
        scores = scores.max(dim=-1).values
    elif attribute_agg_method == "linear":
        scores = linear_agg_module(scores).squeeze(-1)
    elif attribute_agg_method == "none":
        pass
    else:
        raise ValueError("Unknown attribute aggregation method.")

    return scores
