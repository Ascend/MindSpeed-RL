# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.datasets.dict_dataset import trans_batch_to_data_loader, DictDataset

from tests.test_tools.dist_test import DistributedTest


class TestDictDataset(DistributedTest):
    world_size = 1

    def test_DictDataset(self):
        data_dict = {'a': [1, 2, 3]}
        data_len = 3
        dataset = DictDataset(data_dict, data_len)
        assert len(dataset) == data_len, "The __len__ method of the Dataset failed!"
        assert dataset[0] == {'a': 1}, "The __get_item__ method of the Dataset failed!"
        assert dataset[1] == {'a': 2}, "The __get_item__ method of the Dataset failed!"
        assert dataset[2] == {'a': 3}, "The __get_item__ method of the Dataset failed!"


    def test_trans_batch_to_data_loader(self):
        batch = {
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        }
        batch_length = 3
        dataloader = trans_batch_to_data_loader(batch, batch_length)
        assert dataloader.batch_size == batch_length, "The data batch acquisition of the Dataset failed"
        for batch_data in dataloader:
            assert len(batch_data['a']) == batch_length, "The data length acquisition of the Dataset failed!"
            assert len(batch_data['b']) == batch_length, "The data length acquisition of the Dataset failed!"
            assert batch_data['a'].tolist() == [1, 2, 3], "The content of the dataloader is wrong!"
            assert batch_data['b'] == ['x', 'y', 'z'], "The content of the dataloader is wrong!"