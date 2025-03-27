# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import time

import torch
import torch_npu

from mindspeed_rl.utils.utils import get_tune_attention_mask

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
_MAX_DATA_DIM = 5


def get_finetune_data_on_this_tp_rank(data_iterator, parallel_state, reset_attention_mask, tokenizer_padding_side):
    """
    Get the data for fine-tuning on this tensor parallel rank.

    Args:
        data_iterator: Iterable data object.
        parallel_state: ParallelState object.
        reset_attention_mask: Indicating whether to reset the attention mask.
        tokenizer_padding_side: Indicating the padding direction of the tokenizer, which can be "left" or "right".

    Returns:
        tokens: Processed input tokens.
        attention_mask: The attention mask matrix.
    """
    ds = next(data_iterator)
    tokens = ds.get('input_ids').long().cuda(non_blocking=True)
    tokens_shape = tokens.shape
    micro_batch_size = tokens_shape[0]

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, parallel_state.get_tensor_model_parallel_src_rank(),
                                        group=parallel_state.get_tensor_model_parallel_group())

    if parallel_state.get_tensor_model_parallel_rank() == 0:
        via_length = torch.LongTensor([tokens_shape[1]]).cuda(non_blocking=True)
        _broadcast(via_length)
        _broadcast(tokens)
        attention_mask_1d = ds.get('attention_mask').long().cuda(non_blocking=True)
        _broadcast(attention_mask_1d)
        attention_mask = get_tune_attention_mask(attention_mask_1d, reset_attention_mask, tokenizer_padding_side)
    else:
        via_length = torch.empty((1), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(via_length)
        tokens = torch.empty((micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(tokens)
        attention_mask_1d = torch.empty((micro_batch_size, via_length), dtype=torch.int64,
                                        device=torch.cuda.current_device())
        _broadcast(attention_mask_1d)
        attention_mask = get_tune_attention_mask(attention_mask_1d, reset_attention_mask, tokenizer_padding_side)

    return tokens, attention_mask


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        if data[key].dtype != target_dtype:
            raise ValueError('{} has data type {} which '
                             'is different than {}'.format(key, data[key].dtype, target_dtype))


def _build_key_size_numel_dictionaries(keys, data, parallel_state):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if parallel_state.get_tensor_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            if data[key].dim() >= max_dim:
                raise ValueError('you should increase MAX_DATA_DIM')
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.tensor(sizes, dtype=torch.long, device='cuda')
    torch.distributed.broadcast(
        sizes_cuda, parallel_state.get_tensor_model_parallel_src_rank(),
        group=parallel_state.get_tensor_model_parallel_group()
    )

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype, parallel_state):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Args:
        keys: list of keys in the data dictionary to be broadcast
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
        parallel_state: ParallelState object
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data, parallel_state)

    # Pack on rank zero.
    if parallel_state.get_tensor_model_parallel_rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat([data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
    else:
        flatten_data = torch.empty(total_numel, device=torch.cuda.current_device(), dtype=datatype)

    # Broadcast
    torch.distributed.broadcast(
        flatten_data, parallel_state.get_tensor_model_parallel_src_rank(),
        group=parallel_state.get_tensor_model_parallel_group()
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def average_losses_across_data_parallel_group(losses, parallel_state):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=parallel_state.get_data_parallel_group())
    averaged_losses = averaged_losses / \
                      torch.distributed.get_world_size(group=parallel_state.get_data_parallel_group())

    return averaged_losses
