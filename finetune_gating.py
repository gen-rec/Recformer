import json
from argparse import ArgumentParser
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.functional import cross_entropy

import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from utils import AverageMeterSet, Ranker, load_data


def load_config_tokenizer(args, item2id):
    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.attention_window = [64] * 12
    config.max_item_embeddings = args.max_item_len + 1
    config.max_item_len = args.max_item_len
    config.max_token_num = args.max_token_len
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


def train_gating_layer_epoch(mars, gating_layer, dataloader, optimizer, scheduler, epoch, args):
    epoch_losses = []
    gating_layer.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc=f"[Train] Epoch {epoch}")):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with autocast(dtype=torch.bfloat16, enabled=args.bf16), torch.no_grad():
            mars_loss, mars_gating_vector, _ = mars.forward(loss_reduction='none', gating_vector=args.gating_method, **batch)
            # mars_loss (bs, num_attr)
            # mars_gating_vector (bs, hidden_size)

        optimizer.zero_grad()

        weight = gating_layer(mars_gating_vector)  # (bs, num_attr)
        loss = cross_entropy(weight, mars_loss)  # (bs, num_attr)

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.item())
        if wandb_logger is not None:
            wandb_logger.log({"train/batch_loss": loss.item()})

    if wandb_logger is not None:
        wandb_logger.log({"train/epoch_loss": sum(epoch_losses) / len(epoch_losses)})

    return sum(epoch_losses) / len(epoch_losses)


def evaluate_gating_layer(mars, gating_layer, dataloader, args, return_preds=False, mode="val"):
    mars.eval()
    gating_layer.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    all_scores = []
    all_labels = []
    all_weights = []
    total_loss, title_loss, brand_loss, category_loss = 0, 0, 0, 0

    with torch.no_grad():
        for batch, labels in tqdm(dataloader, ncols=100, desc="[Eval]"):
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            labels = labels.to(args.device)
            mars_loss, mars_gating_vector, mars_scores = mars.forward(loss_reduction='none',
                                                                      gating_vector=args.gating_method,
                                                                      **batch)
            weight = gating_layer(mars_gating_vector)  # (bs, num_attr)
            attr_scores = mars_scores * weight.unqueeze(1)  # (bs, num_items, num_attr)
            scores = attr_scores.sum(dim=-1)  # (bs, num_items)

            loss = nn.CrossEntropyLoss()
            title_loss += loss(attr_scores[:, :, 0], labels)
            brand_loss += loss(attr_scores[:, :, 1], labels)
            category_loss += loss(attr_scores[:, :, 2], labels)
            total_loss += loss(scores, labels)

            all_weights.append(weight.detach().clone().cpu())
            all_scores.append(scores.detach().clone().cpu())
            all_labels.append(labels.detach().clone().cpu())

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

    if wandb_logger is not None:
        if mode == "val":
            wandb_logger.log({"valid-loss/title_loss": title_loss / len(dataloader)})
            wandb_logger.log({"valid-loss/brand_loss": brand_loss / len(dataloader)})
            wandb_logger.log({"valid-loss/category_loss": category_loss / len(dataloader)})
            wandb_logger.log({"valid-loss/total_loss": total_loss / len(dataloader)})
            wandb_logger.log({"valid-metrics": average_metrics})
        else:
            wandb_logger.log({"test-loss/title_loss": title_loss / len(dataloader)})
            wandb_logger.log({"test-loss/brand_loss": brand_loss / len(dataloader)})
            wandb_logger.log({"test-loss/category_loss": category_loss / len(dataloader)})
            wandb_logger.log({"test-loss/total_loss": total_loss / len(dataloader)})
            wandb_logger.log({"test-metrics": average_metrics})

    if return_preds:
        all_weights = torch.cat(all_weights, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        all_predictions = torch.topk(all_scores, k=max(args.metric_ks), dim=1).indices
        return average_metrics, all_predictions, all_labels, all_weights

    return average_metrics


def main(args):
    seed_everything(args.seed, workers=True)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item, user2id, id2user = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)
    global tokenizer_glb
    tokenizer_glb = tokenizer

    cur_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    path_corpus = Path(args.data_path)
    path_output = Path(args.output_dir) / f"{path_corpus.name}_{cur_time}"

    path_output.mkdir(exist_ok=True, parents=True)

    global wandb_logger
    wandb_logger = wandb.init(
        project="MarsGating",
        entity="gen-rec",
        name= path_corpus.name + time.strftime("%Y%m%d-%H%M%S"),
        group=path_corpus.name,
        config=vars(args),
    )

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }

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

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(Path(args.pretrain_ckpt) / "stage_2_best.pt", map_location="cpu")
    model.item_embedding = nn.Parameter(pretrain_ckpt["item_embedding"])
    print(model.load_state_dict(pretrain_ckpt, strict=False))
    model.to(args.device)
    model.eval()

    gating_layer = nn.Sequential(
        nn.Linear(config.hidden_size, args.linear_out),
        nn.ReLU(),
        nn.Linear(args.linear_out, 3),
    )
    gating_layer.to(args.device)

    def _initialize(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    gating_layer.apply(_initialize)

    optimizer = optim.Adam(gating_layer.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_target = 0
    patient = 5

    for epoch in range(args.num_train_epochs):
        train_loss = train_gating_layer_epoch(model, gating_layer, train_loader, optimizer, scheduler, epoch, args)
        val_metrics = evaluate_gating_layer(model, gating_layer, dev_loader, args, return_preds=False, mode="val")

        if val_metrics[args.early_stop_metric] > best_target:
            best_target = val_metrics[args.early_stop_metric]
            patient = 5
        else:
            patient -= 1

        if patient == 0:
            break

    test_metrics, predictions, labels, test_weight = evaluate_gating_layer(model, gating_layer, test_loader,
                                                                           args,
                                                                           return_preds=False, mode="test")
    users = list(map(int, test.keys()))
    users = list(map(id2user.get, users))

    predictions = predictions.tolist()
    labels = labels.tolist()

    output = {}
    for user, prediction, label, weight in zip(users, predictions, labels, test_weight):
        prediction = list(map(id2item.get, prediction))
        label = id2item[label]
        weight = weight.tolist()
        output[user] = {"predictions": prediction, "target": label, "weights": weight}

    json.dump(output, open(path_output / "predictions.json", "w"), indent=1, ensure_ascii=False)

def parse_finetune_args():
    parser = ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear_out", type=int, default=256)
    parser.add_argument("--group_name", type=str, default=None)
    # gating
    parser.add_argument("--gating_method", type=str, default="cls", choices=["cls", "mean"])
    # path and file
    parser.add_argument("--pretrain_ckpt", type=str, default=None, required=True)
    parser.add_argument("--data_path", type=Path, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name_or_path", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--dev_file", type=str, default="val.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--user2id_file", type=str, default="umap.json")
    parser.add_argument("--item2id_file", type=str, default="smap.json")
    parser.add_argument("--meta_file", type=str, default="meta_data.json")
    # data process
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=8, help="The number of processes to use for the preprocessing."
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    # model
    parser.add_argument("--max_item_len", type=int, default=50)
    parser.add_argument("--max_token_len", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.05, help="Temperature for softmax.")
    parser.add_argument("--global_attention_type", type=str, default="cls", choices=["cls", "attribute"])

    # train
    parser.add_argument("--early_stop_metric", type=str, default="NDCG@50", help="Metric for early stopping.")
    parser.add_argument("--num_train_epochs", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--finetune_negative_sample_size", type=int, default=-1)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[10, 20, 50], help="ks for Metric@k")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=800)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fix_word_embedding", action="store_true")
    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument(
        "--session_reduce_method", type=str, default="maxsim", choices=["maxsim", "mean", "weightedsim", "topksim"]
    )
    parser.add_argument("--pooler_type", type=str, default="attribute", choices=["attribute", "item", "token", "cls"])
    parser.add_argument("--original_embedding", action="store_true")
    parser.add_argument("--one_step_training", action="store_true")
    parser.add_argument("--session_reduce_topk", type=int, default=None, help="topksim: topk")
    parser.add_argument("--session_reduce_weightedsim_temp", type=float, default=None, help="weightedsim: temp")
    parser.add_argument("--eval_test_batch_size_multiplier", type=int, default=2)
    parser.add_argument("--encode_item_batch_size_multiplier", type=int, default=4)
    parser.add_argument("--random_word", type=str, default=None)
    parser.add_argument("--zero_shot_only", action="store_true")
    parser.add_argument("--attribute_agg_method", type=str, default="mean")
    return parser.parse_args()



if __name__ == "__main__":
    main(parse_finetune_args())

