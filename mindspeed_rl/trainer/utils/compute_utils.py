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
from mindspeed_rl.utils.utils import generate_mask, get_current_dp_range_indexes, extract_from_dict
from mindspeed_rl.trainer.utils.transfer_dock import pad_experience
from mindspeed_rl.utils.utils import mstx_timer_decorator
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.compute import compute_kl_penalty


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


@ray.remote
def apply_kl_penalty(td, experience_count, global_batch_size, guarantee_order, tokenizer, kl_ctrl: AdaptiveKLController, penalty="kl"):
    experience_consumer_stage = "compute_kl"
    experience_columns = ["old_log_prob", "ref_log_prob", "responses", "rm_scores", "response_length"]
    pad_token_id = tokenizer.pad if tokenizer.pad is not None else tokenizer.eod
    sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                  assign_batch_size=global_batch_size) if guarantee_order else None

    while not ray.get(td.all_consumed.remote(experience_consumer_stage)):
        batch_data, index = ray.get(
            td.get_experience.remote(
                experience_consumer_stage, experience_columns, experience_count,
                indexes=sorted_indexes.pop(0) if guarantee_order else None
            )
        )
        if batch_data and index:
            batch_data = pad_experience(batch_data, pad_token_id)
            token_level_scores = batch_data["rm_scores"]
            batch_size = len(index)
            response_mask = generate_mask(batch_data['responses'], batch_data['response_length'])

            kld = compute_kl_penalty(batch_data["old_log_prob"], batch_data["ref_log_prob"],
                                        kl_penalty=penalty)
            kld = kld * response_mask
            beta = kl_ctrl.value

            reward_tensor = torch.zeros_like(batch_data['responses'], dtype=torch.float32)
            for i in range(batch_data['responses'].shape[0]):
                valid_response_length = batch_data['response_length'][i] - 1
                reward_tensor[i, int(valid_response_length.item())] = token_level_scores[i]

            token_level_rewards = reward_tensor - beta * kld

            current_kl = F.masked_mean(kld, mask=response_mask, axis=-1)
            current_kl = torch.mean(current_kl, dim=0).item()

            kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
            token_level_rewards = truncate_rows(token_level_rewards, batch_data['response_length'])
            output = {
                "token_level_rewards": token_level_rewards,
            }
            td.put_experience.remote(data_dict=output, indexes=index)


def compute_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        eos_mask: torch.Tensor,
        gamma: torch.Tensor,
        lam: torch.Tensor
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
    return advantages, returns


@ray.remote
def dynamic_sampling(num_prompt_in_batch, data_num, n_samples_per_prompt, sampling_transfer_dock, transfer_dock, guarantee_order):
    """
    dynamic sampling, filter out all groups with the same metric value in group

    Args:
        num_prompt_in_batch: current train prompt num
        data_num: sampling_transfer_dock data num
        n_samples_per_prompt: n respose per prompt
        sampling_transfer_dock: A data queue object for infer
        transfer_dock: A data queue object for train
        guarantee_order: guarantee order

    Returns:
        num_prompt_in_batch: current train prompt num after filter
    """
    logger = Loggers('dynamic_sampling')
    experience_consumer_stage = 'dynamic_sampling'
    experience_columns = ['prompts', 'prompt_length', 'responses', 'labels', 'response_length',
                          'input_ids', 'rm_scores', 'metric_for_dapo', 'reward_for_dapo']
    sorted_indexes = get_current_dp_range_indexes(experience_count=data_num,
                                                  assign_batch_size=data_num) if guarantee_order else None
    while not ray.get(sampling_transfer_dock.all_consumed.remote(experience_consumer_stage)):
        batch_data, index = ray.get(
            sampling_transfer_dock.get_experience.remote(experience_consumer_stage, experience_columns, experience_count=data_num,
                                                     indexes=sorted_indexes.pop(0) if guarantee_order else None))
        if batch_data and index:
            # filter by metric values
            metric_values = batch_data['metric_for_dapo']
            kept_idx_list = []
            for idx in range(0, data_num, n_samples_per_prompt):
                metric_group = metric_values[idx: idx + n_samples_per_prompt]
                vals = [val.item() for val in metric_group]
                if np.std(metric_group) > 0 or len(metric_group) == 1:
                    num_prompt_in_batch += 1
                    kept_idx_list.extend(list(range(idx, idx + n_samples_per_prompt)))
            logger.info(f"dynamic_sampling: num_prompt_in_batch {num_prompt_in_batch}")
            if not kept_idx_list:
                logger.info(f"dynamic_sampling: kept_idx_list is empty")
                break

            experience_data = extract_from_dict(batch_data, kept_idx_list)
            index_list = ray.get(transfer_dock.prefetch_request_index.remote(len(kept_idx_list)))
            if index_list:
                ray.get(transfer_dock.put_experience.remote(experience_data, index_list))

    return num_prompt_in_batch


def compute_group_norm_advantage_return(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        response_length: torch.Tensor,
        n_sample_per_prompt
):
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
    scores = torch.tensor(
        token_level_rewards,
        dtype=torch.float64,
        device=response_length.device
    )
    scores = scores.reshape(-1, n_sample_per_prompt)
    scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
    scores = scores.reshape(response_length.shape)
    scores = torch.tensor(
        scores,
        dtype=torch.float32,
        device=response_length.device
    )
    new_token_level_rewards = scores.repeat(1, eos_mask.shape[1])
    new_token_level_rewards = new_token_level_rewards * eos_mask
    advantages = deepcopy(new_token_level_rewards)
    returns = deepcopy(advantages)

    return advantages, returns


@ray.remote
@mstx_timer_decorator
def compute_advantage(td, gamma, lam, adv_estimator, experience_count, tokenizer, global_batch_size, guarantee_order, n_sample_per_prompt):
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
    if adv_estimator == "gae":
        experience_columns = ["values", "responses", "token_level_rewards", "response_length"]
    else:
        experience_columns = ["responses", "rm_scores", "response_length"]
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
            response_length = batch_data["response_length"]
            if adv_estimator == "gae":
                token_level_rewards = batch_data["token_level_rewards"]
                values = batch_data["values"]
                advantages, returns = compute_gae_advantage_return(
                    token_level_rewards=token_level_rewards,
                    values=values,
                    eos_mask=response_mask,
                    gamma=gamma,
                    lam=lam
                )
            elif adv_estimator == "group_norm":
                token_level_rewards = batch_data["rm_scores"]
                advantages, returns = compute_group_norm_advantage_return(
                    token_level_rewards=token_level_rewards,
                    eos_mask=response_mask,
                    response_length=response_length,
                    n_sample_per_prompt=n_sample_per_prompt
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
            prompt_length = batch["prompt_length"]
            response_length = batch["response_length"]

            metrics = {
                # score
                "grpo/score/mean": torch.mean(sequence_score).detach().item(),
                "grpo/score/max": torch.max(sequence_score).detach().item(),
                "grpo/score/min": torch.min(sequence_score).detach().item(),
                # response
                "response_length/mean": torch.mean(response_length, dtype=torch.float32).detach().item(),
                "response_length/max": torch.max(response_length).detach().item(),
                "response_length/min": torch.min(response_length).detach().item(),
                # prompt length
                "prompt_length/mean": torch.mean(prompt_length, dtype=torch.float32).detach().item(),
                "prompt_length/max": torch.max(prompt_length).detach().item(),
                "prompt_length/min": torch.min(prompt_length).detach().item(),
            }
            return metrics


def compute_dapo_data_metrics(
        td, experience_count, tokenizer, global_batch_size, guarantee_order
):
    """
    Calculate various metrics for DAPO data

    Args:
        td: A data queue object
        experience_count: Number of experiences to retrieve
        tokenizer: The pre-trained tokenizer
        global_batch_size: The number of global batch size
        guarantee_order: The switch of guarantee order

    Returns:
        Dictionary containing various metric values
    """
    experience_consumer_stage = "dapo_metrics"
    experience_columns = [
        "rm_scores",
        "responses",
        "advantages",
        "returns",
        "prompt_length",
        "response_length",
        "reward_for_dapo"
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
            prompt_length = batch["prompt_length"]
            response_length = batch["response_length"]
            reward_for_dapo = batch["reward_for_dapo"]

            metrics = {
                # score
                "dapo/score/mean": torch.mean(sequence_score).detach().item(),
                "dapo/score/max": torch.max(sequence_score).detach().item(),
                "dapo/score/min": torch.min(sequence_score).detach().item(),

                "response_length/mean": torch.mean(response_length, dtype=torch.float32).detach().item(),
                "response_length/max": torch.max(response_length).detach().item(),
                "response_length/min": torch.min(response_length).detach().item(),
                # prompt length
                "prompt_length/mean": torch.mean(prompt_length, dtype=torch.float32).detach().item(),
                "prompt_length/max": torch.max(prompt_length).detach().item(),
                "prompt_length/min": torch.min(prompt_length).detach().item(),

                "dapo/reward_for_dapo/mean": torch.mean(reward_for_dapo).detach().item(),
            }
            return metrics


def compute_ppo_data_metrics(
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
        "values",
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
            batch = pad_experience(batch, pad_token_id)
            sequence_score = batch["rm_scores"].sum(-1)
            sequence_reward = batch["token_level_rewards"].sum(-1)
            prompt_length = batch["prompt_length"]
            response_length = batch["response_length"]
            advantages = batch['advantages']
            returns = batch['returns']
            values = batch['values']
            response_mask = generate_mask(batch["responses"], batch["response_length"])
            metrics = {
                # score
                'critic/score/mean': torch.mean(sequence_score).detach().item(),
                'critic/score/max': torch.max(sequence_score).detach().item(),
                'critic/score/min': torch.min(sequence_score).detach().item(),
                # reward
                'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
                'critic/rewards/max': torch.max(sequence_reward).detach().item(),
                'critic/rewards/min': torch.min(sequence_reward).detach().item(),
                # adv
                'critic/advantages/mean': F.masked_mean(advantages, response_mask).detach().item(),
                'critic/advantages/max': torch.max(advantages[response_mask]).detach().item(),
                'critic/advantages/min': torch.min(advantages[response_mask]).detach().item(),
                # returns
                'critic/returns/mean': F.masked_mean(returns, response_mask).detach().item(),
                'critic/returns/max': torch.max(returns[response_mask]).detach().item(),
                'critic/returns/min': torch.min(returns[response_mask]).detach().item(),
                # values
                'critic/values/mean': F.masked_mean(values, response_mask).detach().item(),
                'critic/values/max': torch.max(values[response_mask]).detach().item(),
                'critic/values/min': torch.min(values[response_mask]).detach().item(),
                # response length
                'response_length/mean': torch.mean(response_length, dtype=torch.float32).detach().item(),
                'response_length/max': torch.max(response_length).detach().item(),
                'response_length/min': torch.min(response_length).detach().item(),
                # prompt length
                'prompt_length/mean': torch.mean(prompt_length, dtype=torch.float32).detach().item(),
                'prompt_length/max': torch.max(prompt_length).detach().item(),
                'prompt_length/min': torch.min(prompt_length).detach().item(),
            }
            return metrics