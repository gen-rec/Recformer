import random
from typing import Optional

from torch.utils.data import Dataset

from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding


class RecformerTrainDataset(Dataset):
    def __init__(self, user2train, collator: FinetuneDataCollatorWithPadding, items, neg_samples: Optional[int] = None):

        """
        user2train: dict of sequence data, user--> item sequence
        """
        if isinstance(items, list):
            items = set(items)

        # Filter out sessions with only length one
        self.user2train = {k: v for k, v in user2train.items() if len(v) > 1}
        self.user2negatives = {k: list(items - set(v)) for k, v in self.user2train.items() if len(v) > 1}
        self.collator = collator
        self.users = sorted(self.user2train.keys())
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self.user2train[user]

        if self.neg_samples is None:
            return seq

        neg = self.user2negatives[user]
        return seq, neg

    def collate_fn(self, data):
        if self.neg_samples is None:
            return self.collator([{"items": item} for item in data])

        items = []
        negatives = set()

        for seq, neg in data:
            items.append(seq)
            neg = set(neg)

            # Leave only common negatives
            negatives = set.intersection(negatives, neg) if len(negatives) > 0 else neg

        negatives = list(negatives)
        negatives = random.sample(negatives, self.neg_samples) if len(negatives) > self.neg_samples else negatives

        return self.collator([{"items": item} for item in items], negatives)


class RecformerEvalDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, mode, collator: EvalDataCollatorWithPadding):
        self.user2train = user2train
        self.user2val = user2val
        self.user2test = user2test
        self.collator = collator

        if mode == "val":
            self.users = list(self.user2val.keys())
        else:
            self.users = list(self.user2test.keys())

        self.mode = mode

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.user2train[user] if self.mode == "val" else self.user2train[user] + self.user2val[user]
        label = self.user2val[user] if self.mode == "val" else self.user2test[user]

        return seq, label

    def collate_fn(self, data):

        return self.collator([{"items": line[0], "label": line[1]} for line in data])
