# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import pytest
import torch

from tests.test_tools.dist_test import DistributedTest


class TestGetTuneAttentionMask(DistributedTest):
    @pytest.fixture
    def setUp(self):
        self.reset_attention_mask = True
        self.tokenizer_padding_side = "right"
        self.attention_mask_1d = torch.rand(5, 10)

    def test_reset_attention_mask_true(self, setUp):
        from mindspeed_rl.trainer.utils.training import get_tune_attention_mask
        result = get_tune_attention_mask(self.attention_mask_1d, self.reset_attention_mask, self.tokenizer_padding_side)
        assert result.size(0) == 5


class TestCheckDataTypes(DistributedTest):
    def test_same_dtype(self):
        from mindspeed_rl.trainer.utils.training import _check_data_types
        import numpy as np
        data = {
            'key1': np.array([1, 2, 3], dtype=np.int32),
            'key2': np.array([4, 5, 6], dtype=np.int32),
            'key3': np.array([7, 8, 9], dtype=np.int32)
        }
        keys = ['key1', 'key2', 'key3']
        target_dtype = np.int32
        _check_data_types(keys, data, target_dtype)
