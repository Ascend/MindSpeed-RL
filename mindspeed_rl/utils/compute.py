# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from typing import Dict, List, Tuple, Union
import torch

_ParallelState = None
_VocabParallel = None


def set_parallel_state(obj):
    global _ParallelState
    _ParallelState = obj


def get_parallel_state():
    global _ParallelState
    return _ParallelState


def set_vocab_parallel(obj):
    global _VocabParallel
    _VocabParallel = obj


def get_vocab_parallel():
    global _VocabParallel
    return _VocabParallel


def compute_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the log probabilities of the given labels under the given logits.

    In the tensor parallelism case, it takes into account the vocab parallelism and
    performs the necessary adjustments to the labels and logits.

    Args:
        logits: The logits tensor.
        labels: The label tensor.

    Returns:
        Log probabilities.
    """
    vocab_parallel_cross_entropy = get_vocab_parallel()
    labels = labels.clone()
    log_probs = -vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)
    return log_probs


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        mpu = get_parallel_state()

        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        torch.torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        torch.torch.distributed.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        torch.torch.distributed.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks
    
    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)
        
    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)
