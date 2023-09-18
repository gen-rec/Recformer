from collections import namedtuple
from datetime import datetime
from pathlib import Path

import torch
import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from wonderwords import RandomWord

from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset
from optimization import create_optimizer_and_scheduler
from recformer import RecformerForSeqRec, RecformerTokenizer, RecformerConfig, reduce_session, RecformerModelWithPooler
from utils import AverageMeterSet, Ranker, load_data, parse_finetune_args

wandb_logger: wandb.sdk.wandb_run.Run | None = None
tokenizer_glb: RecformerTokenizer | None = None


TokenizedItem = namedtuple("TokenizedItem", ["input_ids", "token_type_ids", "attr_type_ids"])


def load_config_tokenizer(args, item2id):
    config = RecformerConfig.from_pretrained(
        args.model_name_or_path,
        temp=args.temp,
        max_attr_num=3,
        max_attr_length=32,
        max_item_embeddings=51,
        max_token_num=512,
        item_num=len(item2id),
        finetune_negative_sample_size=args.finetune_negative_sample_size,
        pooler_type=args.pooler_type,
        session_reduce_method=args.session_reduce_method,
        session_reduce_topk=args.session_reduce_topk,
        session_reduce_weightedsim_temp=args.session_reduce_weightedsim_temp,
    )

    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    if args.session_reduce_method == "weightedsim" and args.session_reduce_weightedsim_temp is None:
        raise ValueError("session_reduce_weightedsim_temp must be specified when session_reduce_method is weightedsim.")
    if args.session_reduce_method == "topksim" and args.session_reduce_topk is None:
        raise ValueError("session_reduce_topk must be specified when session_reduce_method is topksim.")

    return config, tokenizer


def _par_tokenize_doc(doc):
    item_id, item_attr = doc

    input_ids, token_type_ids, attr_type_ids = tokenizer_glb.tokenize_item(item_attr)

    return item_id, input_ids, token_type_ids, attr_type_ids


@torch.no_grad()
def encode_all_items(model: RecformerModelWithPooler, tokenizer: RecformerTokenizer, tokenized_items, args):
    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    for i in tqdm(
        range(0, len(items), args.batch_size * args.encode_item_batch_size_multiplier),
        ncols=100,
        desc="Encode all items",
    ):

        item_batch = [[item] for item in items[i : i + args.batch_size * args.encode_item_batch_size_multiplier]]

        inputs = tokenizer.batch_encode(item_batch, encode_item=False)

        for k, v in inputs.items():
            inputs[k] = torch.LongTensor(v).to(args.device)

        outputs = model.forward(**inputs)

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


@torch.no_grad()
def evaluate(model: RecformerForSeqRec, dataloader, args):
    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):
        keys = ["attr_type_ids", "token_type_ids", "input_ids"]
        padding_values = [0, 0, model.config.pad_token_id]
        session_len = torch.tensor([len(b["inputs"]) for b in batch])
        batch_concatenated = {key: [] for key in keys}
        for session in batch:
            session_inputs = session["inputs"]
            for item in session_inputs:
                for key in keys:
                    batch_concatenated[key].append(item[key])

        # Move to device
        batch_concatenated = {
            k: pad_sequence(v, batch_first=True, padding_value=padding_value).to(args.device)
            for (k, v), padding_value in zip(batch_concatenated.items(), padding_values)
        }
        batch_concatenated["attention_mask"] = (
            torch.ne(batch_concatenated["input_ids"], model.config.pad_token_id).float().to(args.device)
        )
        labels = labels.to(args.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.bf16):
            scores = model.forward_batch(batch_concatenated, session_len, None)  # (bs, |I|, num_attr, items_max)
            scores = reduce_session(scores, args.session_reduce_method, args.session_reduce_topk)

        assert torch.isnan(scores).sum() == 0, "NaN in scores."

        metrics = ranker.forward(scores, labels)

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    return average_metrics


def train_one_epoch(model: RecformerForSeqRec, dataloader, optimizer, scheduler, args, train_step: int):
    global wandb_logger

    epoch_losses = []

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc="Training")):
        keys = ["attr_type_ids", "token_type_ids", "input_ids"]
        padding_values = [0, 0, model.config.pad_token_id]
        session_len = torch.tensor([len(b["inputs"]) for b in batch])
        batch_concatenated = {key: [] for key in keys}
        labels = torch.tensor([b["labels"] for b in batch])
        for session in batch:
            session_inputs = session["inputs"]
            for item in session_inputs:
                for key in keys:
                    batch_concatenated[key].append(item[key])

        # Move to device
        batch_concatenated = {
            k: pad_sequence(v, batch_first=True, padding_value=padding_value).to(args.device)
            for (k, v), padding_value in zip(batch_concatenated.items(), padding_values)
        }
        batch_concatenated["attention_mask"] = (
            torch.ne(batch_concatenated["input_ids"], model.config.pad_token_id).float().to(args.device)
        )
        labels = labels.to(args.device)

        with autocast(dtype=torch.bfloat16, enabled=args.bf16):
            loss = model.forward_batch(batch_concatenated, session_len, labels)  # (bs, |I|, num_attr, items_max)

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

    seed_everything(42)
    args.device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)
    global tokenizer_glb
    tokenizer_glb = tokenizer

    random_word_generator = RandomWord()
    random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]
    server_random_word_and_date = args.server + "_" + random_word + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Run name:", server_random_word_and_date)

    path_corpus = Path(args.data_path)
    path_output = Path(args.output_dir) / random_word

    try:
        path_output.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        raise FileExistsError(f"Output directory ({path_output}) already exists.")

    global wandb_logger
    wandb_logger = wandb.init(
        project="RecIR",
        entity="gen-rec",
        name=server_random_word_and_date,
        group=path_corpus.name,
        config=vars(args),
        tags=[
            path_corpus.name,
            f"pool_{args.pooler_type}",
            f"reduce_session_{args.session_reduce_method}",
            f"global_attn_{args.global_attention_type}",
        ],
    )

    path_ckpt = path_output / args.ckpt

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: TokenizedItem(input_ids, token_type_ids, attr_type_ids)
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items, config)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items, config)

    train_data = RecformerTrainDataset(train)
    val_data = RecformerEvalDataset(train, val, test, mode="val")
    test_data = RecformerEvalDataset(train, val, test, mode="test")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=finetune_data_collator)
    dev_loader = DataLoader(
        val_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=eval_data_collator
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=eval_data_collator
    )

    model = RecformerForSeqRec(config)
    missing, unexpected = model.model_with_pooler.model.load_state_dict(torch.load(args.pretrain_ckpt), strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.to(args.device)

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.model_with_pooler.model.embeddings.parameters():
            param.requires_grad = False

    item_embeddings = encode_all_items(model.model_with_pooler, tokenizer, tokenized_items, args)
    model.init_item_embedding(item_embeddings)

    model.to(args.device)  # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    test_metrics = evaluate(model, test_loader, args)
    if wandb_logger is not None:
        wandb_logger.log({f"zero-shot/{k}": v for k, v in test_metrics.items()})
    print(f"Test set Zero-shot: {test_metrics}")

    best_target = float("-inf")
    patient = 5

    for epoch in range(args.num_train_epochs):
        if epoch > 0:
            item_embeddings = encode_all_items(model.model_with_pooler, tokenizer, tokenized_items, args)
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
                torch.save(model.state_dict(), path_ckpt)

            else:
                patient -= 1
                if patient == 0:
                    break

    print("Stage 1 Test")
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = evaluate(model, test_loader, args)
    print(f"Test set stage 1: {test_metrics}")

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
                    torch.save(model.state_dict(), path_ckpt)

                else:
                    patient -= 1
                    if patient == 0:
                        break

    print("Test with the best checkpoint.")
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = evaluate(model, test_loader, args)
    print(f"Test set: {test_metrics}")

    if wandb_logger is not None:
        wandb_logger.log({f"test/{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main(parse_finetune_args())
