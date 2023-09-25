import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LongformerForMaskedLM
from wonderwords import RandomWord

from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from recformer import RecformerTokenizer
from recmlm import RecMLMConfig, RecMLMDataModule, RecMLM
from utils import parse_mlm_args, load_data


def _par_tokenize_doc(item, tokenizer):
    item_id, item_attr = item
    input_ids, token_type_ids, attr_type_ids = tokenizer.encode_item(item_attr)
    return item_id, input_ids, token_type_ids, attr_type_ids


def load_config_tokenizer_model(args, item2id):
    config = RecMLMConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)

    config.session_reduce_method = args.session_reduce_method
    config.pooler_type = args.pooler_type
    config.global_attention_type = args.global_attention_type
    config.original_embedding = args.original_embedding

    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config=config)
    model = LongformerForMaskedLM.from_pretrained(args.model_name_or_path, config=config)

    return config, tokenizer, model


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision("medium")
    seed_everything(42)
    args.device = args.accelerator
    args.session_reduce_topk = None
    args.session_reduce_weightedsim_temp = None

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    config, tokenizer, model = load_config_tokenizer_model(args, item2id)

    random_word_generator = RandomWord()
    random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]
    server_random_word_and_date = args.server + "_" + random_word + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path_corpus = Path(args.data_path)
    path_output = Path(args.output_dir) / random_word
    print(f"Output directory: {path_output}")

    # Load datamodule
    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    datamodule = RecMLMDataModule(
        mlm_ratio=args.mlm_ratio,
        tokenizer=tokenizer,
        user2train=train,
        user2val=val,
        user2test=test,
        id2item=id2item,
        item_meta_dict=item_meta_dict,
        batch_size=args.batch_size,
        mlm_batch_multiplier=args.mlm_batch_multiplier,
        num_workers=args.dataloader_num_workers,
    )

    rec_test_loader, rec_val_loader, tokenized_items = load_session_dataset(
        args, item2id, item_meta_dict, path_corpus, test, tokenizer, train, val
    )

    module = RecMLM(
        args=args,
        config=config,
        model=model,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        tokenized_items=tokenized_items,
        rec_val_dataloader=rec_val_loader,
        rec_test_dataloader=rec_test_loader,
        checkpoint_save_path=path_output,
    )

    # logger
    loggers = [
        CSVLogger(save_dir=args.output_dir, name="RecMLM"),
    ]
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project="RecMLM",
            entity="gen-rec",
            group=path_corpus.name,
            name=server_random_word_and_date,
        )
        loggers.append(wandb_logger)

    for logger in loggers:
        logger.log_hyperparams(vars(args))

    # Setup trainer
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=path_output,
            every_n_epochs=1,
            filename="epoch_{epoch}-r10_{rec_metric/Recall@10:.2f}",
            auto_insert_metric_name=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/rec_metric/Recall@10",
            mode="max",
            patience=5,
            verbose=True,
        ),
    ]

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        max_steps=args.max_steps,
        precision=args.precision,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=args.gradient_clip_val,
    )

    config.save_pretrained(path_output)

    trainer.fit(module, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


def load_session_dataset(args, item2id, item_meta_dict, path_corpus, test, tokenizer, train, val):
    doc_tuples = [
        _par_tokenize_doc(doc, tokenizer)
        for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)
    val_data = RecformerEvalDataset(train, val, test, mode="val", collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)
    rec_val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size * args.eval_test_batch_size_multiplier,
        collate_fn=val_data.collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    rec_test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size * args.eval_test_batch_size_multiplier,
        collate_fn=test_data.collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    return rec_test_loader, rec_val_loader, tokenized_items


if __name__ == "__main__":
    argument = parse_mlm_args()
    main(argument)
