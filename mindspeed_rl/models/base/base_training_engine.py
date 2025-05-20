# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import random
from abc import ABC
from typing import Callable, Dict, List
from functools import partial

import torch
from torch.utils.data import DataLoader

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.utils.utils import (
    append_to_dict, generate_mask, generate_position_ids, get_tune_attention_mask
)
from mindspeed_rl.utils.compute import get_parallel_state


class BaseTrainingEngine(ABC):
    """
    Initialize the base trainning engine.

    Args:
        model: The network model to be trained.
        optimizer: The optimizer for updating model parameters (e.g., Adam).
        opt_param_scheduler: The scheduler for optimizer parameters (e.g., learning rate scheduler).
        beta: float = 0 The weight coefficient for KL divergence (used in algorithms like PPO).
        mini_batch_size_per_dp: int = 1  The size of the mini-batch for each data parallel stage.
        epochs: int = 1 The number of training epochs.
        shuffle_mini_batch: bool = False Whether to shuffle the mini-batch data at each epoch.
        stage(str): str = None The training stage identifier (e.g., ray_grpo).
        kl_ctrl: float = 0.1 Adaptive kl ctrl.
        clip_ratio: float = 0.1 The clipping ratio threshold for PPO (limits the policy update range).
        role: str The role of actor in the RLHF frameworker.
        micro_batch_size: int = 1 Micro batch size for actor rollout.
        forward_backward_func: Callable = None The forward-backward function for distributed training.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            model,
            optimizer=None,
            opt_param_scheduler=None,
            beta: float = 0,
            mini_batch_size_per_dp: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            kl_ctrl: float = 0.0,
            clip_ratio: float = 0.1,
            role: str = None,
            micro_batch_size: int = 1,
            forward_backward_func: Callable = None,
            entropy_coeff: float = 0.0,
            kl_penalty: str = "low_var_kl",
            **kwargs):
        self.forward_backward_func = forward_backward_func
        self.micro_batch_size = micro_batch_size
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.beta = beta
        self.mini_batch_size_per_dp = mini_batch_size_per_dp
        self.epochs = epochs
        self.shuffle_mini_batch = shuffle_mini_batch
        self.stage = stage
        self.role = role
        self.kl_ctrl = kl_ctrl
        self.kl_penalty = kl_penalty
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.loss_func: BaseLossFunc = LossFuncFactory.get_instance(self.stage, self.role)
        self.kwargs = kwargs

    @staticmethod
    def _split_batches(batch: Dict, batch_size: int, shuffle_mini_batch: bool, dim: int = 0) -> List[Dict]:
        batches = []
        for key, tensors in batch.items():
            for index, tensor in enumerate(torch.split(tensors, batch_size, dim)):
                if index >= len(batches):
                    batches.append({})
                batches[index][key] = tensor

        if shuffle_mini_batch:
            random.shuffle(batches)
        return batches

    def _forward_backward_batch(self, batch: Dict[str, torch.Tensor], forward_only: bool = False):
        batches = self._split_batches(batch, batch_size=self.micro_batch_size,
                                      shuffle_mini_batch=self.shuffle_mini_batch)
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        self.loss_func.add_loss_meta_info(self.get_loss_meta_func())

        def forward_step(batch_iter, model):
            input_ids, attention_mask, position_ids, process_batch = self._get_forward_batch_info(batch_iter)
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            return output, partial(self.loss_func.compute_loss, batch=process_batch, forward_only=forward_only)

        # batch should be a list of batches inside micro-batches
        losses_reduced = self.forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter(batches),
            model=self.model,
            num_microbatches=n_micro_batch,
            seq_length=seq_len,
            micro_batch_size=self.micro_batch_size,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )
        return losses_reduced

    def get_loss_meta_func(self) -> Dict:
        """
        获取具体的loss计算超参
        :return: loss计算超参。
        """
        return {}

    @staticmethod
    def _get_forward_batch_info(batch_iter):
        batch = next(batch_iter)
        input_ids = batch['input_ids']
        attention_mask_1d = generate_mask(input_ids, batch['prompt_length'] + batch['response_length']).to(
            input_ids.device)
        position_ids = torch.tensor(generate_position_ids(input_ids)).to(input_ids.device)
        attention_mask = get_tune_attention_mask(attention_mask_1d)
        return input_ids, attention_mask, position_ids, batch

    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        模型前反向计算结果后处理
        :param output: 模型前反向计算结果
        :param batch: 参与计算的batch数据
        :return: 模型前向计算结果。
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, data: Dict) -> torch.Tensor:
        """
        模型前向计算
        :param data: 前向计算数据
        :return: 模型前向计算结果。
        """
        for k, v in data.items():
            if v is not None:
                data[k] = v.to(next(self.model[0].parameters()).device)
        for model_module in self.model:
            model_module.eval()
        with torch.no_grad():
            output = self._forward_backward_batch(data, forward_only=True)
            return self.post_process_forward_backward_output(output=output, batch=data)

    def update(self, data: Dict, kl_ctrl=None) -> Dict:
        """
        模型反向更新
        :param data: 反向更新数据
        :param kl_ctrl：KL散度计算controller
        :return: 模型反向计算结果。
        """
        self.kl_ctrl = kl_ctrl
        metrics = {}
        grad_norm_list = []
        for k, v in data.items():
            if v is not None:
                data[k] = v.to(next(self.model[0].parameters()).device)
        mini_batches = self._split_batches(data, batch_size=self.mini_batch_size_per_dp,
                                           shuffle_mini_batch=self.shuffle_mini_batch, dim=0)
        for model_module in self.model:
            model_module.train()
        for _ in range(self.epochs):
            for mini_batch in mini_batches:
                for model_chunk in self.model:
                    model_chunk.zero_grad_buffer()
                self.optimizer.zero_grad()
                metric_micro_batch = self._forward_backward_batch(mini_batch)
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

                if update_successful:
                    data_parallel_world_size = get_parallel_state().get_data_parallel_world_size()
                    increment = self.mini_batch_size_per_dp * data_parallel_world_size
                    self.opt_param_scheduler.step(increment=increment)
                grad_norm_list.append(grad_norm) 

                for metric in metric_micro_batch:
                    append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        grad_norm = sum(grad_norm_list) / len(grad_norm_list)
        metrics["grad_norm"] = grad_norm_list
        return metrics
