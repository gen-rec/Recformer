import json
from functools import partial
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from tqdm.contrib.concurrent import process_map

from collator import PretrainDataCollatorWithPadding
from recformer import RecformerTokenizer


class RecformerDataset(Dataset):
    def __init__(self, dataset: List):
        super().__init__()
        self.dataset = [sess for sess in dataset if len(sess) > 1]
        print(f"Filtered {len(dataset) - len(self.dataset)} sessions")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class RecformerDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: RecformerTokenizer,
        mlm_probability: float,
        train_path: Path,
        val_path: Path,
        item_metadata_path: Path,
        tokenized_cache_save_path: Path,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.train_data = json.load(open(train_path))
        self.val_data = json.load(open(val_path))
        self.item_metadata = json.load(open(item_metadata_path))
        self.tokenized_cache_save_path = tokenized_cache_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.doc_tuples = None
        self.tokenized_items = None

    def prepare_data(self):
        if (self.tokenized_cache_save_path / "doc_tuples.pt").exists():
            self.doc_tuples = torch.load(self.tokenized_cache_save_path / "doc_tuples.pt")
            return

        self.doc_tuples = process_map(
            partial(_tokenize, self.tokenizer),
            self.item_metadata.items(),
            ncols=100,
            desc=f"[Tokenize]",
            max_workers=8,
            chunksize=1000,
        )

        torch.save(self.doc_tuples, self.tokenized_cache_save_path / "doc_tuples.pt")

    def setup(self, stage=None):
        if self.doc_tuples is None:
            self.doc_tuples = torch.load(self.tokenized_cache_save_path / "doc_tuples.pt")

        if self.tokenized_items is None:
            self.tokenized_items = {
                item_id: [input_ids, token_type_ids, attr_type_ids]
                for item_id, input_ids, token_type_ids, attr_type_ids in self.doc_tuples
            }

        if stage == "fit" or stage is None:
            self.train_dataset = RecformerDataset(self.train_data)
            self.val_dataset = RecformerDataset(self.val_data)
        elif stage == "validate":
            self.val_dataset = RecformerDataset(self.val_data)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=PretrainDataCollatorWithPadding(
                tokenizer=self.tokenizer,
                tokenized_items=self.tokenized_items,
                mlm_probability=self.mlm_probability,
            ),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=PretrainDataCollatorWithPadding(
                tokenizer=self.tokenizer,
                tokenized_items=self.tokenized_items,
                mlm_probability=self.mlm_probability,
                is_valid=True,
            ),
            shuffle=False,
        )


def _tokenize(tokenizer, item):
    item_id, item_attr = item

    return item_id, *tokenizer.encode_item(item_attr)
