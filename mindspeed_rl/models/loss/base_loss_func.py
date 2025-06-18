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
                     use_dynamic_bsz=False,
                     actual_micro_batch_size=1,
                     non_loss_data=True) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失函数，子类必须实现。
        :param output: 模型的输出 logits。
        :param batch: 输入数据，包含 responses、attention_mask 等。
        :param forward_only: 是否只进行前向计算。
        :param use_dynamic_bsz: 是否使用动态批量大小,如果使用则根据实际批次大小对每个微批次加权。
        :param actual_micro_batch_size: 配置的微批量大小。
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

    def compute_log_probs_no_pack(self, output: torch.Tensor, batch: Dict[str, torch.Tensor], update=False) -> torch.Tensor:
        responses, logits = self._get_compute_log_probs_input(output, batch)
        log_probs = compute_log_probs(logits, responses)
        if update:
            entropy = vocab_parallel_entropy(logits)
            return log_probs, entropy
        else:
            return log_probs

    def compute_log_probs_with_pack(self, output, batch: Dict[str, torch.Tensor], update=False):
        batch_size = len(output)
        log_probs_list = []
        entropy_list = []
        for i in range(batch_size):
            # 提取单个样本的output和batch
            single_output = output[i].unsqueeze(0)
            single_batch = {key: value[i].unsqueeze(0) for key, value in batch.items()}
            # 逐条计算log_probs
            response, logits = self._get_compute_log_probs_input(single_output, single_batch)
            single_log_probs = compute_log_probs(logits, response)
            log_probs_list.append(single_log_probs)
            if update:
                # 计算entropy
                single_entropy = vocab_parallel_entropy(logits)
                entropy_list.append(single_entropy)

        # 将列表转换为张量
        log_probs = torch.cat(log_probs_list, dim=0)
        if update:
            entropy = torch.cat(entropy_list, dim=0)
            return log_probs, entropy
        else:
            return log_probs

    def compute_log_probs(self, output, batch: Dict[str, torch.Tensor], update=False):
        # output在remove padding开启时type为list，否则为tensor
        
        if isinstance(output, list):
            # output为[b,s,v]维，pack场景下可能单条超长与单条较短的组成单个tensor可能导致内存爆炸,因此特殊处理
            return self.compute_log_probs_with_pack(output, batch, update=update)
        elif isinstance(output, torch.Tensor):
            return self.compute_log_probs_no_pack(output, batch, update=update)
        else:
            raise ValueError(f"Output type {type(output)} is not supported")