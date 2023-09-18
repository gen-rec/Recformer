from argparse import Namespace
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from lightning_dataloader import RecformerDataModule
from recformer import RecformerForPretraining, LitWrapper, RecformerConfig, RecformerTokenizer
from utils import parse_pretrain_args


def tokenize(tokenizer, item):
    item_id, item_attr = item

    return item_id, *tokenizer.encode_item(item_attr)


def main(args: Namespace):
    print(args)
    torch.set_float32_matmul_precision("medium")
    seed_everything(42)

    dataset_path = Path(args.data_path)

    random_word = args.random_word
    random_word_and_date = random_word + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_path = Path(args.output_dir) / random_word

    output_path.mkdir(exist_ok=True, parents=True)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51  # 50 item and 1 for cls
    config.attention_window = [64] * 12
    config.item_num = None
    config.max_token_num = 1024
    config.finetune_negative_sample_size = None
    config.session_reduce_method = "maxsim"
    config.pooler_type = "attribute"
    config.original_embedding = False
    config.global_attention_type = "cls"
    config.session_reduce_topk = None
    config.session_reduce_weightedsim_temp = None

    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    datamodule = RecformerDataModule(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        train_path=dataset_path / args.train_file,
        val_path=dataset_path / args.dev_file,
        item_metadata_path=dataset_path / args.item_attr_file,
        tokenized_cache_save_path=output_path,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    pytorch_model = RecformerForPretraining(config)
    pytorch_model.load_state_dict(torch.load(args.longformer_ckpt))

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    model = LitWrapper(pytorch_model, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps)

    callbacks = [
        ModelCheckpoint(save_top_k=5, monitor="accuracy", mode="max", filename="{epoch}-{accuracy:.4f}"),
        EarlyStopping(
            monitor="val/loss",
            patience=5,
            verbose=True,
            mode="min",
        ),
    ]

    loggers = [
        WandbLogger(
            save_dir=output_path,
            name=random_word_and_date,
            entity="gen-rec",
            project="RecIR-pretrain",
        ),
        CSVLogger(
            save_dir=output_path,
            name=random_word_and_date,
        )
    ]

    for logger in loggers:
        logger.log_hyperparams(args)

    trainer = Trainer(
        accelerator="auto",
        strategy="auto",
        max_epochs=args.max_epochs,
        num_nodes=2,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=args.val_check_interval,
        default_root_dir=args.output_dir,
        log_every_n_steps=25,
        precision="bf16-mixed" if args.bf16 else 32,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(model, datamodule=datamodule)

    if trainer.is_global_zero:
        trainer.save_checkpoint(output_path / "checkpoint.ckpt")
        config.save_pretrained(output_path)
        torch.save(pytorch_model.state_dict(), output_path / "model_state_dict.pt")


if __name__ == "__main__":
    main(parse_pretrain_args())
