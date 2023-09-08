from datetime import datetime
from pathlib import Path

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
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
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
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc="Encode all items"):

            item_batch = [[item] for item in items[i : i + args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)  # (bs, attr_num, 1, hidden_size)

    return item_embeddings


def eval(model, dataloader, args):

    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)  # (bs, |I|, num_attr, items_max)

        # Reduce session length
        if args.session_reduce_method == "maxsim":
            # Replace NaN with -inf
            scores[torch.isnan(scores)] = -float("inf")
            scores = scores.max(dim=-1).values  # (bs, |I|, num_attr)
        elif args.session_reduce_method == "mean":
            scores = scores.nanmean(dim=-1)  # (bs, |I|, num_attr)
        else:
            raise ValueError("Unknown session reduce method.")

        # Reduce attribute
        scores = scores.mean(dim=-1)  # (bs, |I|)

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


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args, step: int):
    global wandb_logger

    epoch_losses = []

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc="Training")):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        if wandb_logger is not None:
            wandb_logger.log({f"train_step_{step}/loss": loss.item()})
            epoch_losses.append(loss.item())

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()

            else:

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()

    if wandb_logger is not None:
        wandb_logger.log({f"train_step_{step}/epoch_loss": sum(epoch_losses) / len(epoch_losses)})


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
    random_word_and_date = random_word + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

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
        name=random_word_and_date,
        group=path_corpus.name,
        config=vars(args),
        tags=[
            path_corpus.name,
            f"pool_{args.pooler_type}",
            f"reduce_session_{args.session_reduce_method}",
            f"global_attn_{args.global_attention}",
        ],
    )

    path_ckpt = path_output / args.ckpt

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
    dev_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=val_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=test_data.collate_fn)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)

    model.init_item_embedding(item_embeddings)

    model.to(args.device)  # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    test_metrics = eval(model, test_loader, args)
    if wandb_logger is not None:
        wandb_logger.log({f"zero-shot/{k}": v for k, v in test_metrics.items()})
    print(f"Test set Zero-shot: {test_metrics}")

    best_target = float("-inf")
    patient = 5

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args, 1)

        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
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

    print("Load best model in stage 1.")
    model.load_state_dict(torch.load(path_ckpt))

    patient = 3

    for epoch in range(args.num_train_epochs):

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args, 2)

        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
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
    test_metrics = eval(model, test_loader, args)
    print(f"Test set: {test_metrics}")

    if wandb_logger is not None:
        wandb_logger.log({f"test/{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main(parse_finetune_args())
