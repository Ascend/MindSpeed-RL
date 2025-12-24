#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Log prob computing strategies.

This module defines strategy classes to compute token-level log probabilities
and entropy for different data modalities. It decouples modality-specific
branches from loss functions, improving extensibility and testability.
"""

from typing import Dict, Tuple

import torch

from mindspeed_rl.utils.compute import compute_log_probs, vocab_parallel_entropy
from mindspeed_rl.utils.pad_process import truncate_prompt_and_pad, truncate_middle_and_pad
from mindspeed_rl.utils.context_parallel import (
    get_tensor_allgather_cp_without_pack,
    get_tensor_allgather_cp_with_pack,
)
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.utils.remove_padding import postprocess_packed_seqs


class LogProbComputer:
    """Interface for computing log probabilities and entropy.

    Subclasses implement `compute` for specific modalities.
    """

    def compute(
        self,
        output: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        skip_entropy: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MultimodalLogProbComputer(LogProbComputer):
    """Log prob computer for multimodal outputs (e.g., ViT + LLM).

    This variant derives logits by truncating and aligning with response tokens
    per prompt/response lengths, then computes token log-probs against
    `responses` rather than `labels`.
    """

    @staticmethod
    def _get_compute_log_probs_input(
        output: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        responses = batch["responses"]
        truncate_lengths = (
            torch.cat(
                [batch["prompt_length"], batch["prompt_length"] + batch["response_length"]],
                dim=1,
            )
            - 1
        )
        logits = truncate_middle_and_pad(responses, output, truncate_lengths)
        return responses, logits

    def compute(self, output, batch, skip_entropy, **kwargs):
        responses, logits = self._get_compute_log_probs_input(output, batch)
        log_probs = compute_log_probs(logits, responses)
        if not skip_entropy:
            entropy = vocab_parallel_entropy(output)
        else:
            entropy = torch.zeros_like(log_probs)
        return log_probs, entropy


class StandardLogProbComputer(LogProbComputer):
    """Default log prob computer for standard text-only outputs.

    Supports optional context-parallel packed sequence post-processing.
    """

    @staticmethod
    def _get_log_probs_remove_prompt_pad(
        logprob: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        responses = batch["responses"]
        truncate_lengths = (
            torch.cat(
                [batch["prompt_length"], batch["prompt_length"] + batch["response_length"]],
                dim=1,
            )
            - 1
        )
        logprob = truncate_prompt_and_pad(responses, logprob, truncate_lengths)
        return logprob

    def compute(self, output, batch, skip_entropy, **kwargs):
        use_remove_padding = kwargs.get("use_remove_padding", False)
        index = kwargs.get("index", None)
        labels = batch["labels"]

        log_probs = compute_log_probs(output, labels)

        cp_size = get_parallel_state().get_context_parallel_world_size()

        if use_remove_padding:
            log_probs_allgather = get_tensor_allgather_cp_with_pack(
                log_probs, cp_size, index
            )
            seqlens_in_batch = kwargs.get("seqlens_in_batch", None)
            cu_seqlens_padded = kwargs.get("cu_seqlens_padded", None)
            seq_len = batch["responses"].shape[-1]
            log_probs = postprocess_packed_seqs(
                log_probs_allgather,
                seqlens_in_batch,
                cu_seqlens_padded,
                seq_len,
                prompt_length=batch["prompt_length"],
            )
            if not skip_entropy:
                entropy = vocab_parallel_entropy(output)
                entropy_allgather = get_tensor_allgather_cp_with_pack(
                    entropy, cp_size, index
                )
                entropy = postprocess_packed_seqs(
                    entropy_allgather,
                    seqlens_in_batch,
                    cu_seqlens_padded,
                    seq_len,
                    prompt_length=batch["prompt_length"],
                )
            else:
                entropy = torch.zeros_like(log_probs)

            return log_probs, entropy

        else:
            log_probs_allgather = get_tensor_allgather_cp_without_pack(
                log_probs, cp_size, index
            )
            log_probs = self._get_log_probs_remove_prompt_pad(log_probs_allgather, batch)
            if not skip_entropy:
                entropy = vocab_parallel_entropy(output)
                entropy_allgather = get_tensor_allgather_cp_without_pack(
                    entropy, cp_size, index
                )
                entropy = self._get_log_probs_remove_prompt_pad(entropy_allgather, batch)
            else:
                entropy = torch.zeros_like(log_probs)

            return log_probs, entropy
