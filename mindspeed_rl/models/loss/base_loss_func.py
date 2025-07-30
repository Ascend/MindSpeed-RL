#!/user/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from mindspeed_rl.utils.compute import compute_log_probs, vocab_parallel_entropy
from mindspeed_rl.utils.pad_process import truncate_prompt_and_pad
from mindspeed_rl.utils.context_parallel import get_tensor_allgather_cp_without_pack, get_tensor_allgather_cp_with_pack
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.utils.remove_padding import postprocess_packed_seqs


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
    def _get_log_probs_remove_prompt_pad(logprob: torch.Tensor, batch: Dict[str, torch.Tensor]): 
        responses = batch['responses']
        truncate_lengths = torch.cat([batch['prompt_length'], batch['prompt_length'] + batch['response_length']], dim=1) - 1
        logprob = truncate_prompt_and_pad(responses, logprob, truncate_lengths)
        return logprob

    def compute_log_probs(self, output, batch: Dict[str, torch.Tensor], skip_entropy=True, **kwargs):
        use_remove_padding = kwargs.get('use_remove_padding', False)
        index = kwargs.get('index', None)
        labels = batch['labels']

        log_probs = compute_log_probs(output, labels)
        
        cp_size = get_parallel_state().get_context_parallel_world_size()

        if use_remove_padding:
            log_probs_allgather = get_tensor_allgather_cp_with_pack(log_probs, cp_size, index)
            seqlens_in_batch = kwargs.get('seqlens_in_batch', None)
            cu_seqlens_padded = kwargs.get('cu_seqlens_padded', None)
            seq_len = batch['responses'].shape[-1]
            log_probs = postprocess_packed_seqs(log_probs_allgather, seqlens_in_batch, cu_seqlens_padded, seq_len, prompt_length=batch['prompt_length'])
            if not skip_entropy:
                entropy = vocab_parallel_entropy(output)
                entropy = postprocess_packed_seqs(entropy, seqlens_in_batch, cu_seqlens_padded, seq_len, prompt_length=batch['prompt_length'])
            else:
                entropy = torch.zeros_like(log_probs)

            return log_probs, entropy

        else:
            log_probs_allgather = get_tensor_allgather_cp_without_pack(log_probs, cp_size, index)
            log_probs = self._get_log_probs_remove_prompt_pad(log_probs_allgather, batch)
            if not skip_entropy:
                entropy = vocab_parallel_entropy(output)
            else:
                entropy = torch.zeros_like(log_probs)

            return log_probs, entropy