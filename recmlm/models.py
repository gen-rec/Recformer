from typing import Union, List

import pytorch_lightning as pl
import torch
from transformers import LongformerForMaskedLM, LongformerConfig


class RecMLMConfig(LongformerConfig):
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
            model: LongformerForMaskedLM,
            learning_rate: float = 1e-5,):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

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
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}