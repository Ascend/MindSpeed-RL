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
