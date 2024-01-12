import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import random

import torch
from transformers import get_linear_schedule_with_warmup, AutoModelForMaskedLM

import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from wonderwords import RandomWord
from torch.utils.data import Dataset

from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from utils import AverageMeterSet, Ranker, load_data, parse_finetune_args, parse_item_mlm_args

wandb_logger: wandb.sdk.wandb_run.Run | None = None
tokenizer_glb: RecformerTokenizer = None

SERVER_URL = "http://129.154.54.103:8080"


def load_config_tokenizer(args, item2id):
    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    config.session_reduce_method = args.session_reduce_method
    config.pooler_type = args.pooler_type
    config.original_embedding = args.original_embedding
    config.global_attention_type = args.global_attention_type
    config.session_reduce_topk = args.session_reduce_topk
    config.session_reduce_weightedsim_temp = args.session_reduce_weightedsim_temp
    config.linear_out = args.linear_out
    config.attribute_agg_method = args.attribute_agg_method

    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    if args.global_attention_type not in ["cls", "attribute"]:
        raise ValueError("Unknown global attention type.")

    if args.session_reduce_method == "weightedsim" and args.session_reduce_weightedsim_temp is None:
        raise ValueError("session_reduce_weightedsim_temp must be specified when session_reduce_method is weightedsim.")
    if args.session_reduce_method == "topksim" and args.session_reduce_topk is None:
        raise ValueError("session_reduce_topk must be specified when session_reduce_method is topksim.")

    return config, tokenizer


def _par_tokenize_doc(doc):
    item_id, item_attr = doc

    input_ids, token_type_ids, attr_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids, attr_type_ids


def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):
    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(
                range(0, len(items), args.batch_size * args.encode_item_batch_size_multiplier),
                ncols=100,
                desc="Encode all items",
        ):

            item_batch = [[item] for item in items[i: i + args.batch_size * args.encode_item_batch_size_multiplier]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            if args.pooler_type != "token":
                item_embeddings.append(outputs.pooler_output.detach())
            else:
                pooler_output = outputs.pooler_output.detach()  # (bs, 1, max_seq_len, hidden_size)
                pooler_output = pooler_output.permute(0, 2, 1, 3)  # (bs, max_seq_len, 1, hidden_size)
                for j in range(pooler_output.shape[0]):
                    output_ = pooler_output[j]  # (max_seq_len, 1, hidden_size)
                    item_embeddings.append(output_)

    if args.pooler_type == "token":
        item_embeddings = torch.nn.utils.rnn.pad_sequence(
            item_embeddings, batch_first=True, padding_value=float("nan")
        )  # (bs, max_seq_len, 1, hidden_size)
    else:
        item_embeddings = torch.cat(item_embeddings, dim=0)  # (bs, attr_num, 1, hidden_size)

    return item_embeddings


def evaluate(model, dataloader, args, return_preds=False):
    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    all_scores = []
    all_labels = []

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad(), autocast(dtype=torch.bfloat16, enabled=args.bf16):
            scores = model(**batch)  # (bs, |I|, num_attr, items_max)

        all_scores.append(scores.detach().clone().cpu())
        all_labels.append(labels.detach().clone().cpu())

        assert torch.isnan(scores).sum() == 0, "NaN in scores."

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2 * i]
            metrics["Recall@%d" % k] = res[2 * i + 1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    if return_preds:
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        all_predictions = torch.topk(all_scores, k=max(args.metric_ks), dim=1).indices
        return average_metrics, all_predictions, all_labels

    return average_metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, args, train_step: int):
    global wandb_logger

    epoch_losses = []

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc="Training")):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with autocast(dtype=torch.bfloat16, enabled=args.bf16):
            loss = model(**batch)

        if torch.any(torch.isnan(loss)):
            continue

        if wandb_logger is not None:
            wandb_logger.log({f"train_step_{train_step}/loss": loss.item()})
            epoch_losses.append(loss.item())

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Update learning rate schedule

    if wandb_logger is not None:
        wandb_logger.log({f"train_step_{train_step}/epoch_loss": sum(epoch_losses) / len(epoch_losses)})


class ItemMLMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class ItemMLMCollator:
    tokenizer: RecformerTokenizer
    tokenized_items: dict
    mlm_ratio: float
    config: RecformerConfig

    def __call__(self, batch_item_ids: List[int]):
        """
        batch_item_ids: list of item ids (batch_size, item ids)
        """
        batch_item_seq = self.sample_train_data(batch_item_ids)  # batch_size * item
        batch_input_id, batch_attention_mask, batch_label_id = self.extract_features(batch_item_seq)  # batch_size * item * seq_len

        padded_batch_input_id = torch.nn.utils.rnn.pad_sequence(
            batch_input_id, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_batch_label_id = torch.nn.utils.rnn.pad_sequence(
            batch_label_id, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return {
            "input_ids": padded_batch_input_id,
            "attention_mask": padded_batch_attention_mask,
            "labels": padded_batch_label_id,
        }

    def sample_train_data(self, batch_item_ids):
        batch_item_seq = []

        for session in batch_item_ids:
            item_seq_len = len(session)
            assert len(session) > 1, "Session length must be greater than 1."

            pos1, pos2 = random.sample(range(item_seq_len), 2)

            batch_item_seq.append(session[min(pos1, pos2):max(pos1, pos2)])
        return batch_item_seq

    def extract_features(self, batch_item_seq):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for item_seq in batch_item_seq:
            input_id_seq = [self.tokenizer.eos_token_id]
            attention_mask_seq = [1]
            label_seq = [-100]
            masked_len = max(1, int(len(item_seq) * self.mlm_ratio))
            masked_item = random.sample(range(len(item_seq)), masked_len)

            for idx, item in enumerate(item_seq):
                input_ids, _, _ = self.tokenized_items[item]
                if idx in masked_item:
                    input_id_seq.extend([self.tokenizer.mask_token_id] * len(input_ids))
                    attention_mask_seq.extend([1] * len(input_ids))
                    label_seq.extend(input_ids)
                else:
                    input_id_seq.extend(input_ids)
                    attention_mask_seq.extend([1] * len(input_ids))
                    label_seq.extend([-100] * len(input_ids))
                assert len(input_id_seq) == len(label_seq) == len(
                    attention_mask_seq), "input_id_seq and label_seq must have the same length."

            input_id_seq = input_id_seq[:self.config.max_token_num-1]
            attention_mask_seq = attention_mask_seq[:self.config.max_token_num-1]
            label_seq = label_seq[:self.config.max_token_num-1]

            input_id_seq.append(self.tokenizer.eos_token_id)
            attention_mask_seq.append(1)
            label_seq.append(-100)

            batch_input_ids.append(torch.LongTensor(input_id_seq))
            batch_attention_mask.append(torch.LongTensor(attention_mask_seq))
            batch_labels.append(torch.LongTensor(label_seq))

        return batch_input_ids, batch_attention_mask, batch_labels


def main(args):
    print(args)

    seed_everything(args.seed, workers=True)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item, user2id, id2user = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)
    global tokenizer_glb
    tokenizer_glb = tokenizer

    if args.random_word is None:
        random_word_generator = RandomWord()
        while True:
            random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]

            if " " in random_word or "-" in random_word:
                continue
            else:
                break
    else:
        random_word = args.random_word
    server_random_word_and_date = args.server + "_" + random_word + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path_corpus = Path(args.data_path)
    path_output = Path(args.output_dir) / random_word

    try:
        path_output.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        raise FileExistsError(f"Output directory ({path_output}) already exists.")

    global wandb_logger
    wandb_logger = wandb.init(
        project="Pretrain_MARS",
        entity="gen-rec",
        name=server_random_word_and_date,
        group=args.group_name or path_corpus.name,
        config=vars(args),
        tags=[
            path_corpus.name,
            f"pool_{args.pooler_type}",
            f"reduce_session_{args.session_reduce_method}",
            f"global_attn_{args.global_attention_type}",
            f"linear_{args.linear_out}",
            f"seed_{args.seed}",
        ],
    )

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }

    # create datset for MLM
    mlm_dataset = list(train.values())
    mlm_train_dataset = mlm_dataset[: int(len(mlm_dataset) * 0.8)]
    mlm_test_dataset = mlm_dataset[int(len(mlm_dataset) * 0.8):]
    mlm_collator = ItemMLMCollator()
    mlm_collator.config = config
    mlm_collator.tokenizer = tokenizer
    mlm_collator.tokenized_items = tokenized_items
    mlm_collator.mlm_ratio = args.mlm_ratio

    mlm_train_dataset = ItemMLMDataset(mlm_train_dataset)
    mlm_test_dataset = ItemMLMDataset(mlm_test_dataset)

    mlm_train_dataloader = DataLoader(
        mlm_train_dataset, batch_size=args.mlm_batch_size, shuffle=True, collate_fn=mlm_collator
    )
    mlm_test_dataloader = DataLoader(
        mlm_test_dataset, batch_size=args.mlm_batch_size, shuffle=True, collate_fn=mlm_collator
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlm_lr, weight_decay=args.mlm_weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=len(mlm_train_dataloader) * args.mlm_epochs
    )

    best_target = float("inf")
    patience = 3
    for epoch in range(args.mlm_epochs):
        model.train()
        for step, batch in enumerate(tqdm(mlm_train_dataloader, ncols=100, desc="MLM Training")):
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            optimizer.zero_grad()
            output = model(**batch)

            if torch.any(torch.isnan(output.loss)):
                continue

            if args.gradient_accumulation_steps > 1:
                loss = output.loss / args.gradient_accumulation_steps
            else:
                loss = output.loss

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(mlm_train_dataloader) - 1:
                optimizer.step()
                model.zero_grad()
                scheduler.step()

            wandb_logger.log({"mlm/train_loss": output.loss})

        model.eval()
        with torch.no_grad():
            epoch_losses = []
            for step, batch in enumerate(tqdm(mlm_test_dataloader, ncols=100, desc="MLM Testing")):
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                output = model(**batch)
                epoch_losses.append(output.loss.item())
            wandb_logger.log({"mlm/test_loss": sum(epoch_losses) / len(epoch_losses)})

            if sum(epoch_losses) / len(epoch_losses) < best_target:
                best_target = sum(epoch_losses) / len(epoch_losses)
                patience = 3
                torch.save(model.state_dict(), path_output / "mlm_best.pt")
            else:
                patience -= 1
                if patience == 0:
                    break

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(path_output / "mlm_best.pt", map_location="cpu")
    del pretrain_ckpt["longformer.embeddings.token_type_embeddings.weight"]
    print(model.load_state_dict(pretrain_ckpt, strict=False))

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode="val", collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(
        val_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=val_data.collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=test_data.collate_fn
    )

    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)

    model.init_item_embedding(item_embeddings)

    model.to(args.device)  # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    test_metrics = evaluate(model, test_loader, args)
    if wandb_logger is not None:
        wandb_logger.log({f"zero-shot/{k}": v for k, v in test_metrics.items()})
    print(f"Test set Zero-shot: {test_metrics}")

    if args.zero_shot_only:
        return

    best_target = float("-inf")
    patient = 5

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, args, 1)

        if (epoch + 1) % args.verbose == 0:
            dev_metrics = evaluate(model, dev_loader, args)
            print(f"Epoch: {epoch}. Dev set: {dev_metrics}")

            if wandb_logger is not None:
                wandb_logger.log({f"dev_step_1/{k}": v for k, v in dev_metrics.items()})

            if dev_metrics["NDCG@10"] > best_target:
                print("Save the best model.")
                best_target = dev_metrics["NDCG@10"]
                patient = 5
                torch.save(model.state_dict(), path_output / "stage_1_best.pt")

            else:
                patient -= 1
                if patient == 0:
                    break

    print("Load best model in stage 1.")
    model.load_state_dict(torch.load(path_output / "stage_1_best.pt"))

    test_metrics = evaluate(model, test_loader, args)
    print(f"Stage-1 Test set: {test_metrics}")
    if wandb_logger is not None:
        wandb_logger.log({f"stage_1_test/{k}": v for k, v in test_metrics.items()})

    if not args.one_step_training:
        patient = 3

        for epoch in range(args.num_train_epochs):

            train_one_epoch(model, train_loader, optimizer, scheduler, args, 2)

            if (epoch + 1) % args.verbose == 0:
                dev_metrics = evaluate(model, dev_loader, args)
                print(f"Epoch: {epoch}. Dev set: {dev_metrics}")

                if wandb_logger is not None:
                    wandb_logger.log({f"dev_step_2/{k}": v for k, v in dev_metrics.items()})

                if dev_metrics["NDCG@10"] > best_target:
                    print("Save the best model.")
                    best_target = dev_metrics["NDCG@10"]
                    patient = 3
                    torch.save(model.state_dict(), path_output / "stage_2_best.pt")

                else:
                    patient -= 1
                    if patient == 0:
                        break

        print("Load best model in stage 2.")
        try:
            model.load_state_dict(torch.load(path_output / "stage_2_best.pt"))
        except FileNotFoundError:
            print("No best model in stage 2. Use the latest model.")

        test_metrics, predictions, labels = evaluate(model, test_loader, args, return_preds=True)
        print(f"Stage-2 Test set: {test_metrics}")

        if wandb_logger is not None:
            wandb_logger.log({f"stage_2_test/{k}": v for k, v in test_metrics.items()})

        users = list(map(int, test.keys()))
        users = list(map(id2user.get, users))

        predictions = predictions.tolist()
        labels = labels.tolist()

        output = {}
        for user, prediction, label in zip(users, predictions, labels):
            prediction = list(map(id2item.get, prediction))
            label = id2item[label]
            output[user] = {"predictions": prediction, "target": label}

        json.dump(output, open(path_output / "predictions.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(parse_item_mlm_args())
