#!/user/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from mindspeed_rl.utils.pad_process import truncate_prompt_and_pad, truncate_middle_and_pad
from mindspeed_rl.utils.utils import is_multimodal
from mindspeed_rl.models.loss.logprob_computer import StandardLogProbComputer, MultimodalLogProbComputer


class BaseLossFunc(ABC):
    def __init__(self):
        self.logprob_computer = MultimodalLogProbComputer() if is_multimodal() else StandardLogProbComputer()

    def add_loss_meta_info(self, meta_info: Dict):
        """
        添加计算loss所需要的超参信息，子类必须实现
        param: meta_info: 超参信息
        """
        pass

    @abstractmethod
    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False,
                     non_loss_data=True,
                     **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失函数，子类必须实现。
        :param output: 模型的输出 logits。
        :param batch: 输入数据，包含 responses、attention_mask 等。
        :param forward_only: 是否只进行前向计算。
        :return: 损失值和统计信息。
        """
        pass

    @staticmethod
    def _get_compute_log_probs_input(output: torch.Tensor, batch: Dict[str, torch.Tensor]):
        if 'responses' not in batch:
            raise ValueError("The responses is None")
        responses = batch['responses']
        truncate_lengths = torch.cat([batch['prompt_length'], batch['prompt_length'] + batch['response_length']], dim=1) - 1
        logits = truncate_middle_and_pad(responses, output, truncate_lengths)
        return responses, logits

    @staticmethod
    def _get_log_probs_remove_prompt_pad(logprob: torch.Tensor, batch: Dict[str, torch.Tensor]):
        responses = batch['responses']
        truncate_lengths = torch.cat([batch['prompt_length'], batch['prompt_length'] + batch['response_length']], dim=1) - 1
        logprob = truncate_prompt_and_pad(responses, logprob, truncate_lengths)
        return logprob

    def compute_log_probs(self, output, batch: Dict[str, torch.Tensor], skip_entropy=True, **kwargs):
        # Strategy-based dispatch; no direct branching here.
        return self.logprob_computer.compute(output, batch, skip_entropy, **kwargs)
