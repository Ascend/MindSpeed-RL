#!/user/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from mindspeed_rl.utils.compute import compute_log_probs, vocab_parallel_entropy
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

    def compute_log_probs(self, output, batch: Dict[str, torch.Tensor], update=False, **kwargs):
        use_remove_padding = kwargs.get('use_remove_padding', False)
        if use_remove_padding:
            seqlens_in_batch = kwargs.get('seqlens_in_batch', None)
            cu_seqlens_padded = kwargs.get('cu_seqlens_padded', None)
            batch_size = seqlens_in_batch.shape[0]
            log_probs_list = []
            entropy_list = []
            for i in range(batch_size):
                start = cu_seqlens_padded[i].item()
                length = seqlens_in_batch[i].item()
                single_output = output[0, start:start + length].unsqueeze(0)  # [1, length, vocab_size]
                single_batch = {key: value[i].unsqueeze(0) for key, value in batch.items()}
                response, logits = self._get_compute_log_probs_input(single_output, single_batch)
                single_log_probs = compute_log_probs(logits, response)
                log_probs_list.append(single_log_probs)
                if update:
                    single_entropy = vocab_parallel_entropy(logits)
                    entropy_list.append(single_entropy)
            log_probs = torch.cat(log_probs_list, dim=0)
            if update:
                entropy = torch.cat(entropy_list, dim=0)
                return log_probs, entropy
            else:
                return log_probs
        else:
            responses, logits = self._get_compute_log_probs_input(output, batch)
            log_probs = compute_log_probs(logits, responses)
            if update:
                entropy = vocab_parallel_entropy(logits)
                return log_probs, entropy
            else:
                return log_probs