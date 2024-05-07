import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn

import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from utils import AverageMeterSet, Ranker, read_json, load_data, parse_finetune_args
from finetune import load_config_tokenizer, _par_tokenize_doc, encode_all_items, evaluate_with_gating

wandb_logger: wandb.sdk.wandb_run.Run | None = None
tokenizer_glb: RecformerTokenizer = None


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


def main(args):
    torch.set_float32_matmul_precision("medium")
    print(args)

    seed_everything(args.seed, workers=True)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item, user2id, id2user = load_data(args)
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
    pretrain_ckpt = torch.load(Path(args.pretrain_ckpt) / "stage_2_best.pt", map_location="cpu")
    model.item_embedding = nn.Parameter(pretrain_ckpt["item_embedding"])
    print(model.load_state_dict(pretrain_ckpt, strict=False))
    model.to(args.device)
    model.eval()

    if args.gating:
        gating_layer = torch.load(Path(args.pretrain_ckpt) / "gating_best.pt", map_location="cpu")
        gating_layer.to(args.device)
        gating_layer.eval()

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    model.to(args.device)  # send item embeddings to device
    if args.gating:
        test_metrics, predictions, labels, weights = evaluate_with_gating(model, gating_layer, test_loader, args,
                                                                 return_preds=True)
    else:
        predictions, targets = predict(model, test_loader, args)

    path = Path(args.pretrain_ckpt).parent / "predictions.json"

    users = list(map(int, test.keys()))
    users = list(map(id2user.get, users))

    predictions = predictions.tolist()
    labels = labels.tolist()

    output = {}
    if args.gating:
        weights = weights.tolist()
        for user, prediction, label, weight in zip(users, predictions, labels, weights):
            prediction = list(map(id2item.get, prediction))
            label = id2item[label]
            output[user] = {"predictions": prediction, "target": label, "weights": weight}
    else:
        for user, prediction, label in zip(users, predictions, labels):
            prediction = list(map(id2item.get, prediction))
            label = id2item[label]
            output[user] = {"predictions": prediction, "target": label}

    json.dump(output, open(path / "predictions.json", "w"), indent=1, ensure_ascii=False)
if __name__ == "__main__":
    main(parse_finetune_args())
