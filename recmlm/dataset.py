import os.path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from tqdm import tqdm

from recformer import RecformerTokenizer

__all__ = ["RecMLMDataModule", "RecMLMDataset"]


class RecMLMDataModule(LightningDataModule):

    def __init__(self, mlm_ratio: float, tokenizer: RecformerTokenizer, user2train: dict[int, list[int]],
                 user2val: dict[int, list[int]] | None, id2item: dict[int, str],
                 item_meta_dict: dict[int, dict[str, str]], batch_size: int, mlm_batch_multiplier: int,
                 num_workers: int = 0):

        super().__init__()
        self.mlm_ratio = mlm_ratio
        self.tokenizer = tokenizer
        self.user2train = user2train
        self.user2val = user2val
        self.id2item = id2item
        self.item_meta_dict = item_meta_dict
        self.batch_size = batch_size * mlm_batch_multiplier
        self.num_workers = num_workers

        self.train_history, self.valid_history, self.test_history = self.get_history()

        if os.path.exists("valid_dataset.pt"):
            self.valid_dataset = torch.load("valid_dataset.pt")
        else:
            self.valid_dataset = self.tokenize_dataset(self.valid_history)
            torch.save(self.valid_dataset, "valid_dataset.pt")

        if os.path.exists("test_dataset.pt"):
            self.test_dataset = self.tokenize_dataset(self.test_history)
        else:
            self.test_dataset = self.tokenize_dataset(self.test_history)
            torch.save(self.test_dataset, "test_dataset.pt")

    def get_history(self):
        train_histories = list(list(self.user2train.values())[:-1])
        valid_histories = list(list(self.user2train.values()))
        test_histories = [self.user2train[user] + self.user2val[user] for user in self.user2val.keys()]

        return train_histories, valid_histories, test_histories

    def tokenize_dataset(self, histories):

        dataset = []

        for history in tqdm(histories, ncols=120, desc=f"Creating dataset"):
            history = [self.item_meta_dict[self.id2item[item_id]] for item_id in history]
            tokenized_history = self.tokenizer(history)
            tokenized_history['label'] = tokenized_history['input_ids'].copy()
            masked_input_ids = []
            for idx, (input_id, token_id) in enumerate(
                    zip(tokenized_history['input_ids'], tokenized_history['token_type_ids'])):
                if token_id == 0:  # bos token
                    is_mask = False
                    masked_input_ids.append(input_id)
                elif token_id == 1:  # item token
                    is_mask = False
                    masked_input_ids.append(input_id)
                elif token_id == 2 and tokenized_history['token_type_ids'][idx - 1] == 1:
                    # mask with mlm_ratio
                    is_mask = True if torch.rand(1) < self.mlm_ratio else False
                    masked_input_ids.append(self.tokenizer.mask_token_id if is_mask else input_id)
                elif token_id == 2:
                    masked_input_ids.append(self.tokenizer.mask_token_id if is_mask else input_id)
                else:
                    raise ValueError(f'Invalid token type id: {token_id}')

            tokenized_history['input_ids'] = masked_input_ids

            dataset.append(tokenized_history)

        return dataset

    def train_dataloader(self):
        print(f"Refresh Train DataLoader")
        train_histories = list(list(self.user2train.values())[:-1])
        train_dataset = self.tokenize_dataset(train_histories)
        return torch.utils.data.DataLoader(
            RecMLMDataset(train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            RecMLMDataset(self.valid_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            RecMLMDataset(self.test_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch_session, pad_to_max: bool = False):
        if pad_to_max:
            max_length = self.tokenizer.config.max_token_num
        else:
            max_length = max([len(session['input_ids']) for session in batch_session])

        batch_input_ids, batch_attention_mask, batch_global_attention_mask, batch_label = [], [], [], []

        for session in batch_session:
            input_ids = session['input_ids']
            attention_mask = session['attention_mask']
            global_attention_mask = session['global_attention_mask']
            label = session['label']

            length_to_pad = max_length - len(input_ids)

            input_ids += [self.tokenizer.pad_token_id] * length_to_pad
            attention_mask += [0] * length_to_pad
            global_attention_mask += [0] * length_to_pad
            label += [-100] * length_to_pad

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_global_attention_mask.append(global_attention_mask)
            batch_label.append(label)

        return {
            'input_ids': torch.tensor(batch_input_ids),
            'attention_mask': torch.tensor(batch_attention_mask),
            'global_attention_mask': torch.tensor(batch_global_attention_mask),
            'label': torch.tensor(batch_label)
        }


class RecMLMDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
