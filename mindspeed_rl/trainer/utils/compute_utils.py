# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from copy import deepcopy

import ray
import torch
import numpy as np

import mindspeed_rl.utils.torch_functional as F
from mindspeed_rl.utils.pad_process import truncate_rows
from mindspeed_rl.utils.utils import generate_mask, get_current_dp_range_indexes
from mindspeed_rl.trainer.utils.transfer_dock import pad_experience
from mindspeed_rl.utils.utils import mstx_timer_decorator


class AdaptiveKLController:
    """
    Adaptive KL trainer described in the paper:
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL trainer."""

    def __init__(self, init_kl_coef):
        self.value = init_kl_coef

    def update(self, current_kl, n_steps):
        pass


def compute_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        eos_mask: torch.Tensor,
        gamma: torch.Tensor,
        lam: torch.Tensor,
):
    """
    Compute advantage

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = F.masked_whiten(advantages, eos_mask)
        advantages = torch.masked_fill(advantages, ~eos_mask, 0)
    return advantages, returns


def compute_group_norm_advantage_return(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor):
    """
    Compute advantage

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    new_token_level_rewards = token_level_rewards.repeat(1, eos_mask.shape[1])
    new_token_level_rewards = new_token_level_rewards * eos_mask
    advantages = deepcopy(new_token_level_rewards)
    returns = deepcopy(advantages)

    return advantages, returns


@ray.remote
@mstx_timer_decorator
def compute_advantage(td, gamma, lam, adv_estimator, experience_count, tokenizer, global_batch_size, guarantee_order):
    """
    Compute the advantage function based on different adv_estimator

    Args:
        td: A data queue object
        gamma: The reward discount factor
        lam: The lambda parameter in advantage estimation
        adv_estimator:  The type of advantage estimator, which can be "gae" or "group_norm"
        experience_count: The number of experiences to retrieve from the experience td
        tokenizer: The pre-trained tokenizer
        global_batch_size: The number of global batch size
        guarantee_order: The switch of guarantee order

    Returns:
        None
    """
    experience_consumer_stage = "compute_advantage"
    experience_columns = ["responses", "token_level_rewards", "response_length"]
    pad_token_id = tokenizer.pad if tokenizer.pad is not None else tokenizer.eod
    sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                  assign_batch_size=global_batch_size) if guarantee_order else None
    while not ray.get(td.all_consumed.remote(experience_consumer_stage)):
        batch_data, index = ray.get(
            td.get_experience.remote(
                experience_consumer_stage, experience_columns, experience_count,  # pad_id=pad_token_id
                indexes=sorted_indexes.pop(0) if guarantee_order else None
            )
        )
        if batch_data and index:
            batch_data = pad_experience(batch_data, pad_token_id) # multiple, tp_size
            response_mask = generate_mask(batch_data["responses"], batch_data["response_length"])
            token_level_rewards = batch_data["token_level_rewards"]

            if adv_estimator == "gae":
                values = batch_data["values"]
                advantages, returns = compute_gae_advantage_return(
                    token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
                )
            elif adv_estimator == "group_norm":
                advantages, returns = compute_group_norm_advantage_return(
                    token_level_rewards=token_level_rewards, eos_mask=response_mask
                )
            else:
                raise NotImplementedError
            advantages = truncate_rows(advantages, batch_data['response_length'])
            returns = truncate_rows(returns, batch_data['response_length'])
            output = {
                "advantages": advantages,
                "returns": returns,
            }
            td.put_experience.remote(data_dict=output, indexes=index)


def get_last_reward(rm_scores, n_sample_batch: int):
    """
    Calculate the final reward value

    Args:
        rm_scores: Raw reward scores
        n_sample_batch: Size of the sample batch

    Returns:
        The standardized final reward value
    """
    reward = rm_scores.reshape(-1, n_sample_batch)
    last_reward = (reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8)
    last_reward = last_reward.reshape(rm_scores.shape)
    return last_reward


def compute_grpo_data_metrics(
        td, experience_count, tokenizer, global_batch_size, guarantee_order
):
    """
    Calculate various metrics for GRPO data

    Args:
        td: A data queue object
        experience_count: Number of experiences to retrieve
        tokenizer: The pre-trained tokenizer
        global_batch_size: The number of global batch size
        guarantee_order: The switch of guarantee order

    Returns:
        Dictionary containing various metric values
    """
    experience_consumer_stage = "grpo_metrics"
    experience_columns = [
        "rm_scores",
        "token_level_rewards",
        "responses",
        "advantages",
        "returns",
        "prompt_length",
        "response_length",
    ]
    pad_token_id = tokenizer.pad if tokenizer.pad is not None else tokenizer.eod
    sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                  assign_batch_size=global_batch_size) if guarantee_order else None
    while not ray.get(td.all_consumed.remote(experience_consumer_stage)):
        batch, index = ray.get(
            td.get_experience.remote(experience_consumer_stage, experience_columns, experience_count,
                                     indexes=sorted_indexes.pop(0) if guarantee_order else None)
        )
        if batch and index:
            batch = pad_experience(batch, pad_token_id) # multiple, tp_size
            sequence_score = batch["rm_scores"].sum(-1)
            sequence_reward = batch["token_level_rewards"].sum(-1)
            prompt_length = batch["prompt_length"]
            response_length = batch["response_length"]

            metrics = {
                # score
                "grpo/score/mean": torch.mean(sequence_score).detach().item(),
                "grpo/score/max": torch.max(sequence_score).detach().item(),
                "grpo/score/min": torch.min(sequence_score).detach().item(),
                # reward
                "grpo/rewards/mean": torch.mean(sequence_reward).detach().item(),
                "grpo/rewards/max": torch.max(sequence_reward).detach().item(),
                "grpo/rewards/min": torch.min(sequence_reward).detach().item(),
                "response_length/mean": torch.mean(response_length, dtype=torch.float32).detach().item(),
                "response_length/max": torch.max(response_length).detach().item(),
                "response_length/min": torch.min(response_length).detach().item(),
                # prompt length
                "prompt_length/mean": torch.mean(prompt_length, dtype=torch.float32).detach().item(),
                "prompt_length/max": torch.max(prompt_length).detach().item(),
                "prompt_length/min": torch.min(prompt_length).detach().item(),
            }
            return metrics
