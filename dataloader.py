from torch.utils.data import Dataset


class RecformerTrainDataset(Dataset):
    def __init__(self, user2train):

        """
        user2train: dict of sequence data, user--> item sequence
        """

        # Filter out sessions with only length one
        self.user2train = {k: v for k, v in user2train.items() if len(v) > 1}
        self.users = sorted(self.user2train.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self.user2train[user]

        return seq


class RecformerEvalDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, mode):
        self.user2train = user2train
        self.user2val = user2val
        self.user2test = user2test

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
