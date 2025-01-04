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
from utils import AverageMeterSet, Ranker, parse_args, load_data

wandb_logger: wandb.sdk.wandb_run.Run | None = None
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

    train, vals, tests, item_meta_dict, item2id, id2item, id2user, join_info = load_data(args)
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

    global wandb_logger
    wandb_logger = wandb.init(
        project="Recformer",
        name=server_random_word_and_date,
        group=args.group or path_corpus.name,
        config=vars(args) | {"dataset_names": join_info["datasets"]},
        tags=[
            path_corpus.name,
            f"seed_{args.seed}",
        ],
    )

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids]
        for item_id, input_ids, token_type_ids in doc_tuples
    }

    print(f"Successfully load {len(tokenized_items)} tokenized items.")
    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    vals_data = []
    tests_data = []
    for val, test in zip(vals, tests):
        vals_data.append(RecformerEvalDataset(train, val, test, mode="val", collator=eval_data_collator))
        tests_data.append(RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_loaders = [DataLoader(val_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=val_data.collate_fn) for val_data in vals_data]
    test_loaders = [DataLoader(test_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=test_data.collate_fn) for test_data in tests_data]

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    print(model.load_state_dict(pretrain_ckpt, strict=False))
    model.to(args.device)

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    print(f"Encoding items.")
    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    model.init_item_embedding(item_embeddings)

    model.to(args.device)  # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    test_metrics = []
    for i, test_loader in enumerate(test_loaders, start=1):
        print(f"Test set {i} / {len(test_loaders)}")
        test_metric = evaluate(model, test_loader, args)
        test_metrics.append(test_metric)

        if wandb_logger is not None:
            wandb_logger.log({f"zero-shot/dataset-{i}/{k}": v for k, v in test_metric.items()})
        print(f"Test set {i} Zero-shot: {test_metric}")

    if args.zero_shot_only:
        return

    best_target = float("-inf")
    patient = args.patience[0]

    user_count = join_info["user_count"]

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, args, 1)

        if (epoch + 1) % args.verbose == 0:
            dev_metrics = []
            for i, val_loader in enumerate(val_loaders, start=1):
                print(f"Dev set {i} / {len(val_loaders)}")
                dev_metric = evaluate(model, val_loader, args)
                dev_metrics.append(dev_metric)
                if wandb_logger is not None:
                    wandb_logger.log({f"dev_step_1/dataset-{i}/{k}": v for k, v in dev_metric.items()})
                print(f"Epoch: {epoch}. Dev set {i} : {dev_metric}")

            ndcg10 = 0.0
            for dev_metric, count in zip(dev_metrics, user_count):
                ndcg10 += dev_metric["NDCG@10"] * (count / sum(user_count))

            if ndcg10 > best_target:
                print("Save the best model.")
                best_target = ndcg10
                torch.save(model.state_dict(), path_output / "stage_1_best.pt")
                patient = args.patience[0]

            else:
                patient -= 1
                if patient == 0:
                    break

    print("Load best model in stage 1.")
    model.load_state_dict(torch.load(path_output / "stage_1_best.pt"))

    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    model.init_item_embedding(item_embeddings)

    # Test
    test_metrics = []
    for i, test_loader in enumerate(test_loaders, start=1):
        print(f"Test set {i} / {len(test_loaders)}")
        test_metric = evaluate(model, test_loader, args)
        test_metrics.append(test_metric)

        if wandb_logger is not None:
            wandb_logger.log({f"stage_1_test/dataset-{i}/{k}": v for k, v in test_metric.items()})
        print(f"Test set {i} Stage-1: {test_metric}")

    if not args.one_step_training:
        patient = args.patience[1]

        for epoch in range(args.num_train_epochs):

            train_one_epoch(model, train_loader, optimizer, scheduler, args, 2)

            if (epoch + 1) % args.verbose == 0:
                dev_metrics = []
                for i, val_loader in enumerate(val_loaders, start=1):
                    print(f"Dev set {i} / {len(val_loaders)}")
                    dev_metric = evaluate(model, val_loader, args)
                    dev_metrics.append(dev_metric)
                    if wandb_logger is not None:
                        wandb_logger.log({f"dev_step_2/dataset-{i}/{k}": v for k, v in dev_metric.items()})
                    print(f"Epoch: {epoch}. Dev set {i} : {dev_metric}")

                ndcg10 = 0.0
                for dev_metric, count in zip(dev_metrics, user_count):
                    ndcg10 += dev_metric["NDCG@10"] * (count / sum(user_count))

                if ndcg10 > best_target:
                    print("Save the best model.")
                    best_target = ndcg10
                    patient = args.patience[1]
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

        # Test
        test_metrics = []
        for i, test_loader in enumerate(test_loaders, start=1):
            print(f"Test set {i} / {len(test_loaders)}")
            test_metric = evaluate(model, test_loader, args)
            test_metrics.append(test_metric)

            if wandb_logger is not None:
                wandb_logger.log({f"stage_2_test/dataset-{i}/{k}": v for k, v in test_metric.items()})
            print(f"Test set {i} Stage-2: {test_metric}")


if __name__ == "__main__":
    main(parse_args())
