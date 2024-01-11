import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from wonderwords import RandomWord

from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from utils import AverageMeterSet, Ranker, load_data, parse_finetune_args

tokenizer_glb: RecformerTokenizer = None


class Trainer:
    def __init__(
        self,
        args: Namespace,
        model: RecformerForSeqRec,
        tokenizer: RecformerTokenizer,
        train_dataset: RecformerTrainDataset,
        val_dataset: RecformerEvalDataset,
        test_dataset: RecformerEvalDataset,
        wandb_logger: wandb.sdk.wandb_run.Run | None = None,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = args.device

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.wandb_logger = wandb_logger

    def train(self):
        self._train_stage_1()

        print("Load best model in stage 1.")
        self.model.load_state_dict(torch.load(self.args.output_dir / "stage_1_best.pt"))  # TODO: change path

        self.test("stage_1_test")

        if self.args.one_step_training:
            return

        self._train_stage_2()

        print("Load best model in stage 2.")
        try:
            self.model.load_state_dict(torch.load(self.args.output_dir / "stage_2_best.pt"))  # TODO: change path
        except FileNotFoundError:
            print("No best model in stage 2. Use the latest model.")

        self.test("stage_2_test")

    def test(
        self,
        test_name: Literal["stage_1_test", "stage_2_test", "zero-shot"],
        return_preds: bool = False,
        log: bool = True,
    ):
        self.model.eval()

        dataloader = self._get_test_dataloader()
        ranker = Ranker(self.args.metric_ks)
        average_meter_set = AverageMeterSet()

        all_scores = []
        all_labels = []

        for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):

            for k, v in batch.items():
                batch[k] = v.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad(), autocast(dtype=torch.bfloat16, enabled=self.args.bf16):
                scores = self.model.forward(**batch)  # (bs, |I|, num_attr, items_max)

            all_scores.append(scores.detach().clone().cpu())
            all_labels.append(labels.detach().clone().cpu())

            assert torch.isnan(scores).sum() == 0, "NaN in scores."

            res = ranker.forward(scores, labels)

            metrics = {}
            for i, k in enumerate(self.args.metric_ks):
                metrics["NDCG@%d" % k] = res[2 * i]
                metrics["Recall@%d" % k] = res[2 * i + 1]
            metrics["MRR"] = res[-3]
            metrics["AUC"] = res[-2]

            for k, v in metrics.items():
                average_meter_set.update(k, v)

        average_metrics = average_meter_set.averages()
        print(f"Test set {test_name}: {average_metrics}")

        if log and self.wandb_logger is not None:
            self.wandb_logger.log({f"{test_name}/{k}": v for k, v in average_metrics.items()})

        if return_preds:
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0).squeeze()
            all_predictions = torch.topk(all_scores, k=max(self.args.metric_ks), dim=1).indices
            return average_metrics, all_predictions, all_labels

        return average_metrics

    def _train_stage_1(self):
        self.model.train()

    def _train_stage_2(self):
        pass

    def _encode_all_items(self):
        self.model.eval()

        item_embeddings = None  # TODO: fill

        self.model.init_item_embedding(item_embeddings)

    def _get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn
        )

    def _get_val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size * self.args.eval_test_batch_size_multiplier,
            collate_fn=self.val_dataset.collate_fn,
        )

    def _get_test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size * self.args.eval_test_batch_size_multiplier,
            collate_fn=self.test_dataset.collate_fn,
        )

    def _calculate_approximate_train_steps(self):
        return int(len(self.train_dataset) / self.args.batch_size) * self.args.num_train_epochs


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
    tokenizer.add_tokens(["<is_item>", "<is_session>"], special_tokens=True)

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


def encode_all_items(
    model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args, item_description=None
):
    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]
    if item_description is not None:
        descriptions = sorted(list(item_description.items()), key=lambda x: x[0])
        descriptions = [ele[1] for ele in descriptions]
    else:
        descriptions = None

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(items), args.batch_size * args.encode_item_batch_size_multiplier),
            ncols=100,
            desc="Encode all items",
        ):

            item_batch = [[item] for item in items[i : i + args.batch_size * args.encode_item_batch_size_multiplier]]
            description_batch = [
                description
                for description in descriptions[i : i + args.batch_size * args.encode_item_batch_size_multiplier]
            ]

            inputs = tokenizer.batch_encode(
                item_batch, encode_item=False, is_item=True, description_batch=description_batch
            )

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


def main(args):
    print(args)

    seed_everything(args.seed, workers=True)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item, user2id, id2user, additional_meta_dict = load_data(args)

    for k in item_meta_dict:
        item_meta_dict[k]["title"] = additional_meta_dict[k]["title"]

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

    wandb_logger = wandb.init(
        project="WWW-Rebuttal",
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

    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in [
            _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
        ]
    }

    # Tokenize item descriptions
    item_description_map = {
        item2id[k1]: "description: " + v1["description"]
        for k1, v1 in additional_meta_dict.items()
        if k1 in item2id.keys()
    }

    tokenized_item_descriptions = {
        item_id: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(desc))[:512]  # TODO: Parameterize max length
        for item_id, desc in tqdm(item_description_map.items(), ncols=100, desc="Tokenize item descriptions")
    }

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode="val", collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    print(model.load_state_dict(pretrain_ckpt, strict=False))
    model.to(args.device)

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    # Code for extra 2 tokens
    print(f"Previous embedding size: {model.longformer.embeddings.word_embeddings.weight.shape}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Current embedding size: {model.longformer.embeddings.word_embeddings.weight.shape}")

    item_embeddings = encode_all_items(
        model.longformer, tokenizer, tokenized_items, args, item_description=tokenized_item_descriptions
    )

    model.init_item_embedding(item_embeddings)

    model.to(args.device)  # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    trainer = Trainer(args, model, train_data, val_data, test_data)
    trainer.test("zero-shot")

    if args.zero_shot_only:
        return

    trainer.train()

    best_target = float("-inf")
    patient = 5

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(
            model.longformer, tokenizer, tokenized_items, args, item_description=tokenized_item_descriptions
        )
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
    main(parse_finetune_args())
