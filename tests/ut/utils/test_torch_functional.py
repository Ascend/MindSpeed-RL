# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import torch

from tests.test_tools.dist_test import DistributedTest


class TestClipByValue(DistributedTest):
    world_size = 1

    def test_clip_by_value(self):
        from mindspeed_rl.utils.torch_functional import clip_by_value
        x = torch.tensor([1.0, 2.0, 3.0])
        tensor_min = torch.tensor([0.0])
        tensor_max = torch.tensor([4.0])
        result = clip_by_value(x, tensor_min, tensor_max)
        assert torch.equal(result, x)


class TestMaskedMean(DistributedTest):
    world_size = 1
    
    def test_masked_mean_1d(self):
        from mindspeed_rl.utils.torch_functional import masked_mean
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        result = masked_mean(values, mask)
        assert result == 2.0


class TestMaskedVar(DistributedTest):
    world_size = 1
    
    def test_masked_var_unbiased_true(self):
        from mindspeed_rl.utils.torch_functional import masked_var
        import numpy as np
        values = np.array([1, 2, 3, 4, 5])
        mask = np.array([1, 1, 1, 1, 1])
        result = masked_var(values, mask, unbiased=True)
        assert result == 2.5


class TestMaskedWhiten(DistributedTest):
    world_size = 1
    
    def test_masked_whiten_shift_mean_true(self):
        from mindspeed_rl.utils.torch_functional import masked_whiten
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        expected_result = torch.tensor([-0.7071, 0.7071, 2.1213, 3.5355])
        result = masked_whiten(values, mask)
        assert torch.allclose(result, expected_result, atol=1e-5)