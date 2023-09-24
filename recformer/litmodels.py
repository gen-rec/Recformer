import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.model = model

    def forward(self, **inputs):
        return self.model.forward(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        self.log(
            "train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True
        )
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        loss = outputs.loss
        correct_num = outputs.cl_correct_num
        total_num = outputs.cl_total_num

        accuracy = 0.0
        if total_num > 0:
            accuracy = correct_num / total_num

        self.log_dict({"val/loss": loss, "val/accuracy": accuracy}, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        torch.save(self.model.state_dict(), f"model_state_dict_step_{self.global_step}.pt")

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
