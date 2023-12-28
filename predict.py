import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from utils import AverageMeterSet, Ranker, read_json

wandb_logger: wandb.sdk.wandb_run.Run | None = None
tokenizer_glb: RecformerTokenizer = None


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
    config.global_attention_type = args.global_attention_type
    config.session_reduce_topk = args.session_reduce_topk
    config.session_reduce_weightedsim_temp = args.session_reduce_weightedsim_temp
    config.linear_out = args.linear_out

    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    if args.global_attention_type not in ["cls", "attribute"]:
        raise ValueError("Unknown global attention type.")

    if args.session_reduce_method == "weightedsim" and args.session_reduce_weightedsim_temp is None:
        raise ValueError("session_reduce_weightedsim_temp must be specified when session_reduce_method is weightedsim.")
    if args.session_reduce_method == "topksim" and args.session_reduce_topk is None:
        raise ValueError("session_reduce_topk must be specified when session_reduce_method is topksim.")

    return config, tokenizer


def load_data(args):

    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))

    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v: k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


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

            item_batch = [[item] for item in items[i : i + args.batch_size * args.encode_item_batch_size_multiplier]]

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


def predict(model, dataloader, args):
    model.eval()

    all_predictions = []
    all_labels = []

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)  # (bs, |I|)

        scores = scores.detach().cpu()
        labels = labels.detach().cpu()

        predictions = torch.topk(scores, k=1, dim=-1).indices

        all_predictions.append(predictions)
        all_labels.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0).tolist()
    all_labels = torch.cat(all_labels, dim=0).tolist()

    return all_predictions, all_labels


def evaluate(model, dataloader, args):
    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)  # (bs, |I|)

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

    return average_metrics


def predict(model, dataloader, args):
    model.eval()

    all_scores = []
    all_labels = []

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)  # (bs, |I|)

        assert torch.isnan(scores).sum() == 0, "NaN in scores."

        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0).tolist()

    predictions = torch.topk(all_scores, k=50, dim=-1).indices.tolist()

    return predictions, all_labels


def train_one_epoch(model, dataloader, optimizer, scheduler, args, train_step: int):
    global wandb_logger

    epoch_losses = []

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc="Training")):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with autocast(dtype=torch.bfloat16, enabled=args.bf16):
            loss = model(**batch)

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


def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument("--pretrain_ckpt", type=str, default=None, required=True)
    parser.add_argument("--data_path", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt", type=str, default="best_model.bin")
    parser.add_argument("--model_name_or_path", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--dev_file", type=str, default="val.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--item2id_file", type=str, default="smap.json")
    parser.add_argument("--meta_file", type=str, default="meta_data.json")

    # data process
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=8, help="The number of processes to use for the preprocessing."
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # model
    parser.add_argument("--temp", type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument("--num_train_epochs", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--finetune_negative_sample_size", type=int, default=1000)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[10, 50], help="ks for Metric@k")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fix_word_embedding", action="store_true")
    parser.add_argument("--verbose", type=int, default=3)

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    print(args)

    seed_everything(args.seed, workers=True)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }

    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)

    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=test_data.collate_fn
    )

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    print(model.load_state_dict(pretrain_ckpt, strict=False))
    model.to(args.device)

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    model.to(args.device)  # send item embeddings to device

    predictions, targets = predict(model, test_loader, args)

    path = Path(args.pretrain_ckpt).parent / "predictions.json"

    json.dump({"predictions": predictions, "targets": targets}, open(path, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
