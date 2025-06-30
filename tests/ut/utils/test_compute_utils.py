# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import pytest

from tests.test_tools.dist_test import DistributedTest


class TestAdaptiveKLController(DistributedTest):
    world_size = 1

    @pytest.fixture
    def setUp(self):
        from mindspeed_rl.trainer.utils.compute_utils import AdaptiveKLController
        self.controller = AdaptiveKLController(init_kl_coef=1.0, target_kl=0.1, horizon=100)

    def test_update_with_positive_error(self, setUp):
        self.controller.update(0.2, 50)
        assert self.controller.value == 1.1

    def test_update_with_negative_error(self, setUp):
        self.controller.update(0.05, 50)
        assert self.controller.value == 0.9


class TestFixedKLController(DistributedTest):
    world_size = 1

    @pytest.fixture
    def setUp(self):
        from mindspeed_rl.trainer.utils.compute_utils import FixedKLController
        self.controller = FixedKLController(init_kl_coef=0.5)

    def test_update_with_valid_input(self, setUp):
        current_kl = 10
        n_steps = 20
        assert self.controller.update(current_kl, n_steps) is None


class TestComputeGaeAdvantageReturn(DistributedTest):
    world_size = 1

    def test_compute_gae_advantage_return(self):
        from mindspeed_rl.trainer.utils.compute_utils import compute_gae_advantage_return
        import torch
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 1.0, 1.5]])
        eos_mask = torch.tensor([[True, True, True]])
        gamma = torch.tensor(0.0)
        lam = torch.tensor(0.95)

        advantages, returns = compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam)

        expected_advantages = torch.tensor([[-1.0, 0.0, 1.0]])
        expected_returns = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.equal(advantages, expected_advantages)
        assert torch.equal(returns, expected_returns)


class TestComputeGroupNormAdvantageReturn(DistributedTest):
    world_size = 1

    def test_compute_group_norm_advantage_return(self):
        import torch
        from mindspeed_rl.trainer.utils.compute_utils import compute_group_norm_advantage_return
        token_level_rewards = torch.tensor([[1.0, 2.0]])
        eos_mask = torch.tensor([[True, True]])
        response_length = torch.tensor([[2], [2]])
        n_sample_per_prompt = 2
        expected_advantages = torch.tensor([[-0.7071, -0.7071], [0.7071, 0.7071]])
        expected_returns = torch.tensor([[-0.7071, -0.7071], [0.7071, 0.7071]])
        advantages, returns = compute_group_norm_advantage_return(token_level_rewards, eos_mask, response_length, n_sample_per_prompt)
        assert torch.allclose(advantages, expected_advantages)
        assert torch.allclose(returns, expected_returns)


class TestGetLastReward(DistributedTest):
    world_size = 1

    def test_get_last_reward(self):
        import torch
        from mindspeed_rl.trainer.utils.compute_utils import get_last_reward
        rm_scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        n_sample_batch = 2
        expected_output = torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071])
        assert torch.allclose(get_last_reward(rm_scores, n_sample_batch), expected_output, atol=1e-4)


class TestKlPenalty(DistributedTest):
    world_size = 1

    def test_kl_penalty(self):
        import torch
        from mindspeed_rl.utils.compute import compute_kl_penalty
        logp = torch.tensor([[0.5, 1.0]])
        ref_logp = torch.tensor([[1.0, 0.5]])
        expected_output_kl = torch.tensor([[-0.5000, 0.5000]])
        expected_output_abs = torch.tensor([[0.5000, 0.5000]])
        expected_output_mse = torch.tensor([[0.1250, 0.1250]])
        expected_output_low = [[0.14872121810913086, 0.10653066635131836]]
        assert torch.allclose(compute_kl_penalty(logp, ref_logp, 'kl'), expected_output_kl)
        assert torch.allclose(compute_kl_penalty(logp, ref_logp, 'abs'), expected_output_abs)
        assert torch.allclose(compute_kl_penalty(logp, ref_logp, 'mse'), expected_output_mse)
        assert compute_kl_penalty(logp, ref_logp, 'low_var_kl').tolist() == expected_output_low
        