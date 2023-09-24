import argparse
from pprint import pprint
from typing import Union, List

import pytorch_lightning as pl
import torch
from transformers import LongformerForMaskedLM

from finetune import encode_all_items, evaluate
from recformer import RecformerConfig, RecformerForSeqRec, RecformerTokenizer


class RecMLMConfig(RecformerConfig):
    def __init__(
            self,
            attention_window: Union[List[int], int] = 64,
            sep_token_id: int = 2,
            token_type_size: int = 4,  # <s>, key, value, <pad>
            max_token_num: int = 2048,
            max_item_embeddings: int = 32,  # 1 for <s>, 50 for items
            max_attr_num: int = 12,
            max_attr_length: int = 8,
            **kwargs,
    ):
        super().__init__(attention_window=attention_window, sep_token_id=sep_token_id, **kwargs)
        self.token_type_size = token_type_size
        self.max_token_num = max_token_num
        self.max_item_embeddings = max_item_embeddings
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length


class RecMLM(pl.LightningModule):

    def __init__(
            self,
            args: argparse.Namespace,
            config: RecMLMConfig,
            model: LongformerForMaskedLM,
            tokenizer: RecformerTokenizer,
            learning_rate: float,
            tokenized_items: dict,
            rec_val_dataloader: torch.utils.data.DataLoader,
            rec_test_dataloader: torch.utils.data.DataLoader):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.learning_rate = learning_rate

        self.tokenizer = tokenizer
        self.tokenized_items = tokenized_items
        self.rec_valid_dataloader = rec_val_dataloader
        self.rec_test_dataloader = rec_test_dataloader

    def forward(self, input_ids, attention_mask, global_attention_mask, label, *args, **kwargs):
        outputs = self.model(
            *args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=label,
            **kwargs,
        )
        return outputs

    def on_train_start(self):
        self.evaluate_rec()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.evaluate_rec()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_end(self) -> None:
        self.evaluate_rec()

    @torch.no_grad()
    def evaluate_rec(self):
        recformer = RecformerForSeqRec(self.config)
        state_dict = self.model.longformer.state_dict()

        del state_dict["embeddings.token_type_embeddings.weight"]

        recformer.longformer.load_state_dict(state_dict, strict=False)
        recformer.to(self.args.device)

        item_embeddings = encode_all_items(model=recformer.longformer, tokenizer=self.tokenizer,
                                           tokenized_items=self.tokenized_items, args=self.args)
        recformer.init_item_embedding(item_embeddings)
        test_metrics = evaluate(recformer, self.rec_test_dataloader, self.args)

        recformer.to(torch.device("cpu"))
        del recformer

        pprint(test_metrics)

        for logger in self.trainer.loggers:
            logger.log_metrics({f"rec_metric/{k}": v for k, v in test_metrics.items()})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
