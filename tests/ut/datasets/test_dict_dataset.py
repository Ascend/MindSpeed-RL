# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import unittest
from unittest.mock import MagicMock

import torch

from mindspeed_rl.datasets.dict_dataset import trans_batch_to_data_loader, DictDataset


class TestDictDataset(unittest.TestCase):
    def test_DictDataset(self):
        data_dict = {'a': [1, 2, 3]}
        data_len = 3
        dataset = DictDataset(data_dict, data_len)
        self.assertEqual(len(dataset), data_len)
        self.assertEqual(dataset[0], {'a': 1})
        self.assertEqual(dataset[1], {'a': 2})
        self.assertEqual(dataset[2], {'a': 3})

    def test_trans_batch_to_data_loader(self):
        batch = {
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        }
        batch_length = 3
        dataloader = trans_batch_to_data_loader(batch, batch_length)
        self.assertEqual(dataloader.batch_size, batch_length)
        for batch_data in dataloader:
            self.assertEqual(len(batch_data['a']), batch_length)
            self.assertEqual(len(batch_data['b']), batch_length)
            self.assertEqual(batch_data['a'].tolist(), [1, 2, 3])
            self.assertEqual(batch_data['b'], ['x', 'y', 'z'])