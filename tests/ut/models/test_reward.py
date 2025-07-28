# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import math
from unittest.mock import MagicMock, patch

import pytest

import torch
from torch.utils.data import DataLoader

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.models.reward import Reward
from tests.test_tools.dist_test import DistributedTest


class TestReward(DistributedTest):
    is_dist_test = False

    @pytest.fixture
    def setUp(self):
        self.model = MagicMock()
        self.forward_backward_func = MagicMock()

        self.reward = Reward(
            model=self.model,
            beta=0.1,
            stage="train",
            forward_backward_func=self.forward_backward_func
        )

    def test_initialization(self, setUp):
        assert math.isclose(self.reward.beta, 0.1, rel_tol=1e-5)
        assert self.reward.stage == 'train'
        assert self.reward.role == 'reward'
        assert self.reward.forward_backward_func == self.forward_backward_func

    def test_post_process_forward_backward_output(self, setUp):
        output = torch.tensor([1.0, 2.0, 3.0])
        batch = {"input": torch.tensor([4.0, 5.0, 6.0])}

        processed_output, processed_batch = self.reward.post_process_forward_backward_output(output, batch)

        assert torch.equal(processed_output, output)
        assert processed_batch == batch

    @patch.object(BaseTrainingEngine, "forward")
    def test_compute_rm_score(self, mock_forward, setUp):
        mock_forward.return_value = torch.tensor([1.0, 2.0, 3.0])
        data_loader = DataLoader([1, 2, 3])

        rm_score = self.reward.compute_rm_score(data_loader)
        assert torch.equal(rm_score, torch.tensor([1.0, 2.0, 3.0]))
        mock_forward.assert_called_once_with(data_loader)
