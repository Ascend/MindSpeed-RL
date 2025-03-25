#!/user/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from mindspeed_rl.utils.compute import compute_log_probs
from mindspeed_rl.utils.pad_process import truncate_middle_and_pad


class BaseLossFunc(ABC):
    def __init__(self):
        pass

    def add_loss_meta_info(self, meta_info: Dict):
        """
        添加计算loss所需要的超参信息，子类必须实现
        param: meta_info: 超参信息
        """
        pass

    @abstractmethod
    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False, non_loss_data=True) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失函数，子类必须实现。
        :param output: 模型的输出 logits。
        :param batch: 输入数据，包含 responses、attention_mask 等。
        :param forward_only
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

    def compute_log_probs(self, output: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        responses, logits = self._get_compute_log_probs_input(output, batch)
        log_probs = compute_log_probs(logits, responses)
        return log_probs
