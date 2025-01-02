from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from wonderwords import RandomWord

from ModelMerger.merger import ModelMerger
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from utils import AverageMeterSet, Ranker, parse_args, load_data

tokenizer_glb: RecformerTokenizer = None


def load_config_tokenizer(args, item2id):
    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = args.max_token_num
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    return config, tokenizer


def _par_tokenize_doc(doc):
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids


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

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)  # .cpu()

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
        metrics["loss"] = res[-1]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    if return_preds:
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        all_predictions = torch.topk(all_scores, k=max(args.metric_ks), dim=1).indices
        return average_metrics, all_scores, all_predictions, all_labels

    return average_metrics


def main(args):
    print(args)

    seed_everything(args.seed, workers=True)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item, id2user = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)

    global tokenizer_glb
    tokenizer_glb = tokenizer

    random_word_generator = RandomWord()
    while True:
        random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]
        if " " in random_word or "-" in random_word:
            continue
        else:
            break
    server_random_word_and_date = random_word + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path_corpus = Path(args.data_path)
    path_output = Path(args.output_dir) / random_word

    try:
        path_output.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        raise FileExistsError(f"Output directory ({path_output}) already exists.")

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples
    }

    print(f"Successfully load {len(tokenized_items)} tokenized items.")
    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)

    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=test_data.collate_fn
    )

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    models_ckpt = [torch.load(model_ckpt, map_location="cpu") for model_ckpt in args.models]

    # Only leave keys that are in models
    model_keys = set(models_ckpt[0].keys())
    pretrain_ckpt = {k: v for k, v in pretrain_ckpt.items() if k in model_keys}

    merger = ModelMerger(models=models_ckpt, base_model=pretrain_ckpt)

    results = []
    for weight in range(1, 11):
        weight /= 10
        print(f"Searching for the best merge with weight {weight}")

        merged_state_dict = merger.merge(args.merge_type, weights=weight, density=args.density)

        print(model.load_state_dict(merged_state_dict, strict=False))
        model.to(args.device)

        print(f"Encoding items.")
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        model.to(args.device)  # send item embeddings to device

        test_metrics = evaluate(model, test_loader, args)
        results.append({"weight": weight, **test_metrics})

        print({k: round(v, 4) for k, v in test_metrics.items()})

    results = pd.DataFrame(results)
    output = uuid4().hex[:8]

    results.to_csv(f"{output}.csv", index=False)
    print(f"{output}.csv")


if __name__ == "__main__":
    main(parse_args())
