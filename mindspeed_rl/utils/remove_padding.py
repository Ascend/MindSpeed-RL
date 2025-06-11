# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from typing import Tuple    

import numpy as np
import torch
import torch_npu


def preprocess_packed_seqs(
    input_ids: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    tp_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs variable-length sequences from a batch into a single contiguous tensor for efficient processing.

    Parameters:
        input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.
        attention_mask_1d (torch.Tensor): Binary mask tensor of shape (batch_size, seq_len) where
            each entry indicates valid token positions (1) vs padding (0). dtype should be torch.int or torch.bool.
        tp_size (int): Alignment factor for packing; sequences are padded so that their lengths are
            are multiples of this size.

    Returns:
        input_ids_packed (torch.Tensor): Tensor of shape (1, pack_length) with all valid tokens packed sequentially.
        position_ids_packed (torch.Tensor): Tensor of shape (1, pack_length) containing positional
            indices within each padded sequence block.
        seqlens_in_batch (torch.Tensor): 1D int32 tensor of shape (batch_size,) with original
            sequence lengths (number of valid tokens per sample).
        cu_seqlens_padded (torch.Tensor): 1D int32 tensor of shape (batch_size+1,) containing
            cumulative padded sequence lengths, used for indexing into the packed tensor.

    Raises:
        ValueError: If input_ids and attention_mask_1d have incompatible shapes.
    """
    batch_size, seq_len = input_ids.shape
    if attention_mask_1d.shape != (batch_size, seq_len):
        raise ValueError("attention_mask_1d must have shape (batch_size, seq_len) matching input_ids")

    # Compute actual sequence lengths per sample
    seqlens_in_batch = attention_mask_1d.sum(dim=1, dtype=torch.int32)
    # Compute padding needed to align lengths to tp_size
    pad_size = (tp_size - (seqlens_in_batch % tp_size)) % tp_size


    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    # Cumulative lengths without and with padding
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # Total packed length after padding
    pack_length = int(seqlens_in_batch_padded.sum().item())
    input_ids_packed = torch.zeros(pack_length, dtype=input_ids.dtype, device=input_ids.device)
    # Copy valid tokens sequentially
    for i in range(batch_size):
        start = cu_seqlens_padded[i].item()
        length = seqlens_in_batch[i].item()
        input_ids_packed[start:start + length] = input_ids[i, :length]

    # Generate position IDs within each padded segment
    position_ids_packed = torch.zeros(pack_length, dtype=torch.int32, device=input_ids.device)
    for i in range(batch_size):
        start = cu_seqlens_padded[i].item()
        end = cu_seqlens_padded[i + 1].item()
        position_ids_packed[start:end] = torch.arange(
            end - start, dtype=torch.int32, device=input_ids.device
        )

    return (
        input_ids_packed.unsqueeze(0),
        position_ids_packed.unsqueeze(0),
        seqlens_in_batch,
        cu_seqlens_padded
    )


def postprocess_packed_seqs(
    output: torch.Tensor,
    seqlens_in_batch: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seq_len: int
) -> torch.Tensor:
    """
    Unpacks a packed output tensor back into the original batch shape, restoring padding.

    Parameters:
        output (torch.Tensor): Packed tensor of shape (1, pack_length, ...), typically the model output.
        seqlens_in_batch (torch.Tensor): 1D int32 tensor of original sequence lengths, shape (batch_size,).
        cu_seqlens_padded (torch.Tensor): 1D int32 tensor of cumulative padded lengths, shape (batch_size+1,).
        batch_size (int): Original batch size.
        seq_len (int): Maximum sequence length (including padding) for the output reconstruction.

    Returns:
        output_new (torch.Tensor): Tensor of shape (batch_size, seq_len, ...), with original outputs
            in the first seqlens_in_batch positions and zeros for padding positions.

    Raises:
        ValueError: If output tensor does not have expected batch dimension of 1.
    """
    if output.shape[0] != 1:
        raise ValueError("Expected output tensor to have shape[0] == 1 (packed batch dimension)")

    # Prepare new output with padding
    batch_size = seqlens_in_batch.shape[0]
    full_shape = [batch_size, seq_len] + list(output.shape[2:])
    output_new = torch.zeros(full_shape, dtype=output.dtype, device=output.device)

    for i in range(batch_size):
        start = cu_seqlens_padded[i].item()
        length = seqlens_in_batch[i].item()
        output_new[i, :length] = output[0, start:start + length]

    return output_new

