# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from torch.utils.data import Dataset, DataLoader


class DictDataset(Dataset):
    def __init__(self, data_dict, data_len):
        self.data_dict = data_dict
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        res = {}
        for key, value in self.data_dict.items():
            res[key] = value[idx]
        return res


def trans_batch_to_data_loader(batch, batch_length):
    batch_dataset = DictDataset(batch, batch_length)
    return DataLoader(batch_dataset, batch_size=batch_length, shuffle=False)  # shuffle must be False