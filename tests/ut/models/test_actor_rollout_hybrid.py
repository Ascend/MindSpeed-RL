# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import math
from unittest.mock import MagicMock, patch

import pytest

import torch
from torch.utils.data import DataLoader

from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.actor import Actor
from tests.test_tools.dist_test import DistributedTest


class TestActorRolloutHybrid(DistributedTest):

    @pytest.fixture
    def setUp(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.opt_param_scheduler = MagicMock()
        self.inference_model = MagicMock()
        self.sharding_manager = MagicMock()
        self.generate_config = GenerateConfig({"infer_tensor_parallel_size": 8})
        self.forward_backward_func = MagicMock()

        self.actor_rollout = ActorRolloutHybrid(
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            inference_model=self.inference_model,
            sharding_manager=self.sharding_manager,
            beta=0.1,
            mini_batch_size=32,
            epochs=3,
            shuffle_mini_batch=True,
            stage="ray_grpo",
            generate_config=self.generate_config,
            clip_ratio=0.2,
            forward_backward_func=self.forward_backward_func
        )

    def test_initialization(self, setUp):
        assert self.actor_rollout.generate_config == self.generate_config
        assert isinstance(self.actor_rollout.train_actor, Actor)
        assert self.actor_rollout.train_actor.mini_batch_size == 32
        assert self.actor_rollout.train_actor.epochs == 3
        assert self.actor_rollout.train_actor.shuffle_mini_batch == True
        assert self.actor_rollout.train_actor.stage == 'ray_grpo'
        assert math.isclose(self.actor_rollout.train_actor.beta, 0.1, rel_tol=1e-5)
        assert math.isclose(self.actor_rollout.train_actor.clip_ratio, 0.2, rel_tol=1e-5)
        assert self.actor_rollout.inference_actor == self.inference_model
        assert self.actor_rollout.sharding_manager == self.sharding_manager

    def test_generate_sequences(self, setUp):
        prompts_list = [[1, 2, 3], [4, 5, 6]]
        expected_responses = torch.tensor([[7, 8, 9], [10, 11, 12]])
        self.inference_model.generate_sequences.return_value = (expected_responses,)

        responses = self.actor_rollout.generate_sequences(prompts_list)
        assert torch.equal(responses, expected_responses)
        self.inference_model.generate_sequences.assert_called_once_with(prompts_list)

    @patch.object(Actor, "compute_log_prob")
    def test_compute_log_prob(self, mock_compute_log_prob, setUp):
        mock_compute_log_prob.return_value = torch.tensor([1.0, 2.0, 3.0])
        data_loader = DataLoader([1, 2, 3])

        log_prob = self.actor_rollout.compute_log_prob(data_loader)
        assert torch.equal(log_prob, torch.tensor([1.0, 2.0, 3.0]))
        mock_compute_log_prob.assert_called_once_with(data_loader)

    @patch.object(Actor, "update_actor")
    def test_update_actor(self, mock_update_actor, setUp):
        mock_update_actor.return_value = {"loss": torch.tensor(0.5)}
        data_loader = DataLoader([1, 2, 3])
        kl_ctrl = 0.01

        result = self.actor_rollout.update_actor(data_loader, kl_ctrl)
        assert result == {"loss": torch.tensor(0.5)}
        mock_update_actor.assert_called_once_with(data_loader, kl_ctrl)
