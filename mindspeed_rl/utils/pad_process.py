# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from typing import Dict, List, Tuple, Union
from torch import Tensor

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from mindspeed_rl.utils.utils import mstx_timer_decorator


@mstx_timer_decorator
def remove_padding_and_split_to_list(responses: torch.Tensor, eos_token_id: int, pad_token_id: int, to_list: bool = False) -> List[
    torch.Tensor]:
    output = []
    for i in range(responses.shape[0]):
        response = responses[i]
        nonzeros = torch.nonzero(response == pad_token_id, as_tuple=False)
        if len(nonzeros) != 0:
            first_pad_index = nonzeros[0][0]
        else:
            first_pad_index = len(response)
        if pad_token_id == eos_token_id:
            response = response[:first_pad_index + 1]
        else:
            response = response[:first_pad_index]
        if to_list:
            response = response[:-1].cpu().numpy().tolist()
        output.append(response)
    return output


def pad_multiple(data_list: List[Tensor], pad_id: Union[float, int], multiple: int = 1) -> Tensor:
    padded = pad_sequence(data_list, batch_first=True, padding_value=pad_id)
    max_len = padded.size(1)
    target_len = ((max_len + multiple - 1) // multiple) * multiple
    padded = F.pad(padded, (0, target_len - max_len), value=pad_id)

    return padded


def truncate_middle_and_pad(responses, input_tensor, truncate_lengths, pad_value=0.0):
    """
    input_tensor: Tensor of shape (mbs, seq_len, vocab_size)
    truncate_lengths: Tensor of shape (mbs, 2), where truncate_lengths[i, 0] is the start index to keep,
                      and truncate_lengths[i, 1] is the end index to keep (exclusive).
    pad_value: Value to use for padding (default is 0.0)
    """

    mbs, seq_len, vocab_size = input_tensor.shape

    # Ensure truncate_lengths is within valid range
    truncate_lengths = torch.clamp(truncate_lengths, 0, seq_len)

    # Calculate the new lengths after truncation
    new_lengths = truncate_lengths[:, 1] - truncate_lengths[:, 0]  # (mbs,)

    # Find the maximum length after truncation
    max_new_len = responses.shape[-1]

    # Initialize the output tensor with padding values
    output_tensor = torch.full((mbs, max_new_len, vocab_size), pad_value, dtype=input_tensor.dtype,
                               device=input_tensor.device)

    # Fill the output tensor with truncated values
    for i in range(mbs):
        start_idx = truncate_lengths[i, 0].item()  # Start index to keep
        end_idx = truncate_lengths[i, 1].item()  # End index to keep (exclusive)
        new_len = new_lengths[i].item()  # New length after truncation

        # Copy the middle part of the row to the output tensor
        output_tensor[i, :new_len] = input_tensor[i, start_idx:end_idx]

    return output_tensor


def truncate_rows(tensor, index_tensor, left_pad=False):
    """
    tensor: 二维 Tensor，形状为 (mbs, seq_len)
    index_tensor: 二维 Tensor，形状为 (mbs, 1)，表示每一行截断的位置
    """
    mbs, seq_len = tensor.shape
    truncated_tensors = []

    for i in range(mbs):
        # 获取当前行的截断索引
        trunc_idx = index_tensor[i].item()
        # 截断当前行
        if left_pad:
            truncated_row = tensor[i, -trunc_idx:].cpu()
        else:
            truncated_row = tensor[i, :trunc_idx].cpu()
        # 将截断后的行添加到列表中
        truncated_tensors.append(truncated_row)

    return truncated_tensors
