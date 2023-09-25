import argparse
from pathlib import Path
from pprint import pprint
from typing import Union, List

import pytorch_lightning as pl
import torch
from transformers import LongformerForMaskedLM, get_constant_schedule_with_warmup

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
        rec_test_dataloader: torch.utils.data.DataLoader,
        checkpoint_save_path: Path,
    ):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.learning_rate = learning_rate

        self.tokenizer = tokenizer
        self.tokenized_items = tokenized_items
        self.rec_valid_dataloader = rec_val_dataloader
        self.rec_test_dataloader = rec_test_dataloader

        self.checkpoint_save_path = checkpoint_save_path

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
        self.evaluate_rec(is_test=True)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        metrics = self.evaluate_rec(is_test=False)

        self.log_dict(
            {f"val/rec_metric/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            torch.save(self.model.state_dict(), self.checkpoint_save_path / f"longformer_epoch_{self.current_epoch}.pt")

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        label = batch["label"]
        outputs = self(input_ids, attention_mask, global_attention_mask, label)
        loss = outputs.loss
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_end(self) -> None:
        metrics = self.evaluate_rec(is_test=True)

        for logger in self.loggers:
            logger.log_metrics({f"test/rec_metric/{k}": v for k, v in metrics.items()})

    @torch.no_grad()
    def evaluate_rec(self, is_test: bool):
        recformer = RecformerForSeqRec(self.config)
        state_dict = self.model.longformer.state_dict()

        del state_dict["embeddings.token_type_embeddings.weight"]

        recformer.longformer.load_state_dict(state_dict, strict=False)
        recformer.to(self.args.device)
        recformer.eval()

        item_embeddings = encode_all_items(
            model=recformer.longformer, tokenizer=self.tokenizer, tokenized_items=self.tokenized_items, args=self.args
        )
        recformer.init_item_embedding(item_embeddings)

        if is_test:
            metrics = evaluate(recformer, self.rec_test_dataloader, self.args)
        else:
            metrics = evaluate(recformer, self.rec_valid_dataloader, self.args)

        recformer.to(torch.device("cpu"))
        del recformer

        pprint(metrics)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
