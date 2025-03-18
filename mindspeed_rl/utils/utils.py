# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import sys

import time
import random
from typing import Dict, List

import omegaconf
import numpy as np
import torch
import torch_npu
from torch import Tensor


def synchronize_time():
    """Synchronize training start time across all distributed processes."""
    cur_time = time.time()
    start_time_tensor = torch.tensor([cur_time], dtype=torch.float, device='cuda')
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    min_time = time.asctime(time.localtime(start_time_tensor.item()))
    return min_time


def generate_position_ids(input_ids: Tensor):
    position_ids_list = []
    for i in range(input_ids.size(0)):
        position_ids = list(range(input_ids.size(1)))
        position_ids_list.append(position_ids)
    return position_ids_list


def generate_mask(data_pad: Tensor, seq_lengths: Tensor):
    seq_masks = []
    max_length = data_pad.size(1)
    seq_lengths = seq_lengths.flatten().tolist()
    for seq_length in seq_lengths:
        seq_pad_length = max_length - seq_length
        seq_mask = torch.ones(max_length)
        if seq_pad_length > 0:
            seq_mask[-seq_pad_length:] = 0
        seq_masks.append(seq_mask.numpy().tolist())
    return torch.tensor(seq_masks, dtype=torch.int64)


def get_tune_attention_mask(attention_mask_1d, reset_attention_mask=True, tokenizer_padding_side="right"):
    """
    Generate the final attention mask based on the input parameters and attention mask.

    Args:
        reset_attention_mask: Indicating whether to reset the attention mask.
        tokenizer_padding_side: Indicating the padding direction of the tokenizer, which can be "left" or "right".
        attention_mask_1d: A 1D attention mask tensor.

    Returns:
        The final attention mask tensor.
    """
    micro_batch_size, seq_length = attention_mask_1d.size()
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    if tokenizer_padding_side == "left":
        attention_mask = torch.tril(
            torch.ones(seq_length, seq_length, device=attention_mask_1d.device, dtype=torch.bool)).view(1, 1,
                                                                                                        seq_length,
                                                                                                        seq_length)
        attention_mask_tran = attention_mask_1d.view(seq_length, 1, -1)
        attention_mask = attention_mask.masked_fill((attention_mask_tran < 0.5).view(-1, 1, 1, seq_length), value=0)
    else:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def num_floating_point_operations(args, batch_size):
    """
    Calculate the number of floating-point operations for a given model configuration and batch size.
    
    Args:
        args (object): An object containing various model configuration parameters, including:
            - kv_channels: The number of key-value channels in attention layers.
            - num_attention_heads: The number of attention heads in the model.
            - hidden_size: The dimensionality of the hidden layers.
            - num_layers: The number of hidden layers in the model.
            - seq_length: The length of input sequences.
            - group_query_attention: A boolean indicating whether to use group query attention.
            - num_experts: The number of experts in the Mixture of Experts (MoE) layer.
            - moe_router_topk: The number of top experts to route inputs to in MoE.
            - swiglu: A boolean indicating whether to use SwiGLU activation.
            - padded_vocab_size: The size of the padded vocabulary.
        batch_size (int): The number of samples processed in one batch.

    Returns:
        int: The total number of floating-point operations required for the model with the given configuration and batch size.
    """
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
            12
            * batch_size
            * args.seq_length
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                    (
                            (
                                    1
                                    + (args.num_query_groups / args.num_attention_heads)
                                    + (args.seq_length / args.hidden_size)
                            ) * query_projection_to_hidden_size_ratio
                    )
                    # MLP.
                    + (
                            (args.ffn_hidden_size / args.hidden_size)
                            * num_experts_routed_to
                            * gated_linear_multiplier
                    )
                    # Logit.
                    + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
            )
    )


def get_batch_metrices_mean(metrics_list: List[Dict]) -> Dict[str, Tensor]:
    """
    Calculate the mean of each metric across a list of metric dictionaries.

    Args:
        metrics_list: A list of dictionaries, where each dictionary contains metrics as key-value pairs.

    Returns:
        metrics_mean: A dictionary where each key is a metric name and 
                      each value is the mean of that metric across all batches.
    """
    batch_metrics = {}
    for metrics in metrics_list:
        if metrics:
            append_to_dict(batch_metrics, metrics)
    metrics_mean = {key: torch.tensor(value).mean() for key, value in batch_metrics.items()}
    return metrics_mean


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def parse_args_from_config(config):
    # model configs
    # Parsing utils parameters.
    for key, value in config.items():  # config is transformed into a dict
        if isinstance(value, omegaconf.listconfig.ListConfig):
            sys.argv.append(f"--{key.replace('_', '-')}")
            for i in value:
                sys.argv.append(f"{i}")
        elif isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key.replace('_', '-')}")
        elif value is None:
            continue
        else:
            sys.argv.append(f"--{key.replace('_', '-')}={value}")
