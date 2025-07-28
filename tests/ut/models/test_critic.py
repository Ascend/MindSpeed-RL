# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import math
from unittest.mock import MagicMock, patch

import pytest

import torch
from torch.utils.data import DataLoader

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.models.critic import Critic
from tests.test_tools.dist_test import DistributedTest


class TestCritic(DistributedTest):
    is_dist_test = False

    @pytest.fixture
    def setUp(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.opt_param_scheduler = MagicMock()
        self.forward_backward_func = MagicMock()

        self.critic = Critic(
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            beta=0.1,
            mini_batch_size_per_dp=32,
            epochs=1,
            shuffle_mini_batch=True,
            stage="ray_ppo",
            clip_ratio=0.5,
            forward_backward_func=self.forward_backward_func
        )

    def test_initialization(self, setUp):
        assert self.critic.mini_batch_size_per_dp == 32
        assert self.critic.shuffle_mini_batch is True
        assert self.critic.stage == 'ray_ppo'
        assert math.isclose(self.critic.beta, 0.1, rel_tol=1e-5)
        assert math.isclose(self.critic.clip_ratio, 0.5, rel_tol=1e-5)
        assert self.critic.role == 'critic'
        assert self.critic.forward_backward_func is not None

    def test_get_loss_meta_func(self, setUp):
        meta_info = self.critic.get_loss_meta_func()
        assert 'cliprange_value' in meta_info

    @patch.object(BaseTrainingEngine, "forward")
    def test_compute_values(self, mock_forward, setUp):
        mock_forward.return_value = (torch.tensor([1.0, 2.0, 3.0]), {"meta": "data"})
        data_loader = DataLoader([1, 2, 3])

        log_prob, meta_info = self.critic.compute_values(data_loader)
        assert torch.equal(log_prob, torch.tensor([1.0, 2.0, 3.0]))
        assert meta_info == {"meta": "data"}
        mock_forward.assert_called_once_with(data_loader)

    def test_post_process_forward_backward_output(self, setUp):
        output = torch.tensor([1.0, 2.0, 3.0])
        batch = {"input": torch.tensor([4.0, 5.0, 6.0])}
        processed_output, processed_batch = self.critic.post_process_forward_backward_output(output, batch)

        assert torch.equal(processed_output, output)
        assert processed_batch == batch

    @patch.object(BaseTrainingEngine, "update")
    def test_update_critic(self, mock_update, setUp):
        mock_update.return_value = {"loss": torch.tensor(0.5)}
        data_loader = DataLoader([1, 2, 3])
        kl_ctrl = 0.01

        result = self.critic.update_critic(data_loader, kl_ctrl)
        assert result == {"loss": torch.tensor(0.5)}
        mock_update.assert_called_once_with(data_loader, kl_ctrl)
