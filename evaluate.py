from argparse import ArgumentParser

from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from finetune import evaluate, load_config_tokenizer, _par_tokenize_doc, encode_all_items
from recformer import RecformerForSeqRec
from utils import load_data


def parse_eval_args():
    parser = ArgumentParser()

    parser.add_argument("--metric_ks", nargs="+", type=int, default=[10, 50], help="ks for Metric@k")
    # path and file
    parser.add_argument("--model_name_or_path", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--data_path", type=Path, default=None, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best_model.bin")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--dev_file", type=str, default="val.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--item2id_file", type=str, default="smap.json")
    parser.add_argument("--meta_file", type=str, default="meta_data.json")
    # model
    parser.add_argument(
        "--session_reduce_method", type=str, default="maxsim", choices=["maxsim", "mean", "weightedsim", "topksim"]
    )

    parser.add_argument("--pooler_type", type=str, default="attribute", choices=["attribute", "item", "token", "cls"])
    parser.add_argument("--original_embedding", action="store_true")
    parser.add_argument("--global_attention_type", type=str, default="cls", choices=["cls", "attribute"])
    parser.add_argument("--finetune_negative_sample_size", type=int, default=-1)

    parser.add_argument("--session_reduce_topk", type=int, default=None, help="topksim: topk")
    parser.add_argument("--session_reduce_weightedsim_temp", type=float, default=None, help="weightedsim: temp")
    # data process
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=8, help="The number of processes to use for the preprocessing."
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size_multiplier", type=int, default=1)
    parser.add_argument("--encode_item_batch_size_multiplier", type=int, default=4)

    return parser.parse_args()


def main(args):
    print(args)

    seed_everything(42)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)

    path_corpus = Path(args.data_path)
    print(f"Tokenizing {path_corpus}")
    doc_tuples = [
        _par_tokenize_doc(doc, tokenizer) for doc in
        tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)
    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, collate_fn=test_data.collate_fn
    )

    model_ckpt = Path(args.checkpoint_dir) / args.ckpt
    print(f"Loading checkpoint from {model_ckpt}")
    model = RecformerForSeqRec(config)
    model.load_state_dict(torch.load(model_ckpt))
    model.to(args.device)
    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    model.init_item_embedding(item_embeddings)
    model.load_state_dict(torch.load(model_ckpt))

    test_metrics = evaluate(model, test_loader, args, args.checkpoint_dir)
    print("Test metrics: ", test_metrics)


if __name__ == "__main__":
    main(args=parse_eval_args())
