import torch
import numpy as np
from mindspeed_rl.utils.compute import get_parallel_state


def get_batch_on_this_cp_rank(megatron_config, batch, actual_seq_len):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    index = None
    if megatron_config.context_parallel_size <= 1:
        return batch

    if (megatron_config.context_parallel_algo == 'megatron_cp_algo' and megatron_config.reset_attention_mask and megatron_config.cp_attention_mask_type == 'causal'):
        batch, index = _get_batch_on_this_cp_rank_in_megatron_cp_eod_padding(megatron_config, batch, actual_seq_len)
    elif megatron_config.context_parallel_algo == 'megatron_cp_algo':
        batch, index = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif megatron_config.context_parallel_algo == 'ulysses_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)

    return batch, index


def _get_batch_on_this_cp_rank_in_megatron_cp_eod_padding(megatron_config, batch, actual_seq_len):
    def get_index(batched_actual_seq_len, cp_size, cp_rank):
        full_indices = list(range(len(batched_actual_seq_len) * batched_actual_seq_len[0][-1]))
        batched_index = []
        start = 0
        offset = 0
        for actual_seq_len in batched_actual_seq_len:
            for end in actual_seq_len:
                end = end + offset
                chunk_size = (end - start) // (2 * cp_size)
                batched_index.extend(full_indices[start + cp_rank * chunk_size: start + (cp_rank + 1) * chunk_size])
                batched_index.extend(full_indices[end - (cp_rank + 1) * chunk_size: end - cp_rank * chunk_size])
                start = end
            offset += actual_seq_len[-1]

        return torch.tensor(batched_index, device='npu')

    cp_rank = get_parallel_state().get_context_parallel_rank()
    cp_size = get_parallel_state().get_context_parallel_world_size()

    actual_seq_len_lst = (torch.tensor(actual_seq_len) * get_ring_degree(megatron_config)).tolist()

    batched_index = [actual_seq_len_lst]

    index = get_index(batched_index, cp_size, cp_rank)

    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            bsz = val.shape[0]
            val = val.view(-1, *val.shape[seq_dim + 1:])
            val = val.index_select(0, index)
            val = val.view(bsz, -1, *val.shape[seq_dim + 1:])
        
        batch[key] = val

    return batch, index


def _get_batch_on_this_cp_rank_in_megatron_cp(batch):
    cp_rank = get_parallel_state().get_context_parallel_rank()
    cp_size = get_parallel_state().get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            batch[key] = val

    return batch, index


def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
    cp_rank = get_parallel_state().get_context_parallel_rank()
    cp_size = get_parallel_state().get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def get_ring_degree(megatron_config):
    cp_size = megatron_config.context_parallel_size
    if cp_size == 1:
        return 1
    if megatron_config.context_parallel_algo == 'megatron_cp_algo':
        return cp_size
    elif megatron_config.context_parallel_algo == 'ulysses_cp_algo':
        return 1
    else:
        ring_degree, remainder = divmod(cp_size, megatron_config.ulysses_degree_in_cp)
        if not (ring_degree > 1 and remainder == 0):
            raise ValueError(
                f"--ulysses-degree-in-cp ({megatron_config.ulysses_degree_in_cp}) must be devisible by --context-parallel-size ({cp_size})"
            )
        return ring_degree


def allgather_tensor_cp_group(cp_tensor, cp_size):
    output_list = [torch.empty_like(cp_tensor) for _ in range(cp_size)]
    torch.distributed.all_gather(output_list, cp_tensor.detach(), group=get_parallel_state().get_context_parallel_group())
    output_list[get_parallel_state().get_context_parallel_rank()] = cp_tensor
    output_all_cp = torch.cat(output_list, dim=1)
    return output_all_cp


def get_tensor_allgather_cp_with_pack(cp_tensor, cp_size, index):
    # cp_tensor allgather
    output_all_cp = allgather_tensor_cp_group(cp_tensor, cp_size)
    # when use ring cp, the index is not none. Need to restore the output order based on the index. 
    if index is not None:
        # index allgather
        index_list = [torch.empty_like(index) for _ in range(cp_size)]
        torch.distributed.all_gather(index_list, index, group=get_parallel_state().get_context_parallel_group())
        index_all_cp = torch.cat(index_list, dim=0).cpu().numpy()
        index_all_cp_argsort = np.argsort(index_all_cp)
        
        output_order_restored = output_all_cp[:, index_all_cp_argsort]
        output = output_order_restored
    else:
        output = output_all_cp

    return output


def get_tensor_allgather_cp_without_pack(cp_tensor, cp_size, index):
    # cp_tensor allgather
    output_all_cp = allgather_tensor_cp_group(cp_tensor, cp_size)
    if index is not None:
        # Step1 get index and argsort it for select
        index_list = []
        for cp_rank in range(cp_size):
            index_cp = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
            index_list.append(index_cp)
        index_all_cp = torch.cat(index_list, dim=0)
        index_all_cp_argsort = np.argsort(index_all_cp)
        
        # Step2 chunk output by dim=1, and restore the output as index
        output_all_cp_chunk = torch.chunk(output_all_cp, 2 * cp_size, dim=1)                  
        output_order_restored = [output_all_cp_chunk[i] for i in index_all_cp_argsort]
        output = torch.cat(output_order_restored, dim=1)
    else:
        output = output_all_cp
    return output