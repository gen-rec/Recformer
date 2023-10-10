from argparse import ArgumentParser

from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from finetune import eval, _par_tokenize_doc, encode_all_items, load_data
from recformer import RecformerForSeqRec, RecformerConfig, RecformerTokenizer


def parse_eval_args():
    parser = ArgumentParser()
    # evaluate
    parser.add_argument("--checkpoint_dir", type=str, required=True)

    # path and file
    parser.add_argument("--data_path", type=str, default=None, required=True)
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
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[10, 20, 50], help="ks for Metric@k")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fix_word_embedding", action="store_true")
    parser.add_argument("--verbose", type=int, default=3)

    return parser.parse_args()


def __par_tokenize_doc(doc, tokenizer_glb):
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids


def main(args):
    print(args)

    seed_everything(42)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    path_corpus = Path(args.data_path)
    print(f"Tokenizing {path_corpus}")
    doc_tuples = [
        __par_tokenize_doc(doc, tokenizer) for doc in
        tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids]
        for item_id, input_ids, token_type_ids in doc_tuples
    }
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)
    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, collate_fn=test_data.collate_fn
    )

    stage_1_model_ckpt = Path(args.checkpoint_dir) / Path("stage_1_best_model.pt")
    stage_2_model_ckpt = Path(args.checkpoint_dir) / Path("stage_2_best_model.pt")
    print(f"Loading stage 1 model from {stage_1_model_ckpt}")
    stage_1_model = RecformerForSeqRec(config)
    load_result = stage_1_model.load_state_dict(torch.load(stage_1_model_ckpt), strict=False)
    print(load_result)
    stage_1_model.to(args.device)
    item_embeddings = encode_all_items(stage_1_model.longformer, tokenizer, tokenized_items, args)
    del stage_1_model

    print(f"Loading stage 2 model from {stage_2_model_ckpt}")
    stage_2_model = RecformerForSeqRec(config)
    load_result = stage_2_model.load_state_dict(torch.load(stage_2_model_ckpt), strict=False)
    print(load_result)
    stage_2_model.to(args.device)
    stage_2_model.init_item_embedding(item_embeddings)

    test_metrics = eval(stage_2_model, test_loader, args)
    print("Test metrics: ", test_metrics)


if __name__ == "__main__":
    main(args=parse_eval_args())
