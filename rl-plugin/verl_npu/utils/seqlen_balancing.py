import copy
import heapq
from itertools import chain
from typing import List

import torch
from torch import distributed as dist
from verl_npu.patch_util import NPUPatchHelper
from verl.utils import seqlen_balancing
from verl.utils.device import get_device_name


def ceildiv(a, b):
    return -(a // -b)


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


def heapq_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    equal_part_num = len(seqlen_list) // k_partitions

    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)

    # Initialize the heap: each group maintains [current sum, number of elements, group index, elements in the group]
    groups = [[0, 0, i, []] for i in range(k_partitions)]
    heapq.heapify(groups)

    partitions = []
    for seqlen, i in sorted_seqlen:
        current_group = heapq.heappop(groups)
        current_group[3].append(i)
        current_group[0] += seqlen
        current_group[1] += 1
        if equal_size:
            if current_group[1] < equal_part_num:
                heapq.heappush(groups, current_group)
            else:
                partitions.append(current_group[3])
        else:
            heapq.heappush(groups, current_group)

    partitions.extend([group[3] for group in groups])

    if equal_size:
        for i, partition in enumerate(partitions):
            if len(partition) * k_partitions != len(seqlen_list):
                raise ValueError(f"Partition {i} has {len(partition)} items, expected {len(seqlen_list) // k_partitions}")
    return partitions


def heapq_partition_with_max(seqlen_list: List[int], k_partitions: int, max_token_len: int):
    # 初始化
    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)

    # 初始化堆：每个组维护 [当前和, 元素数量, 组编号, 组内元素]
    groups = [[0, 0, i, []] for i in range(k_partitions)]
    group_num = len(groups)
    heapq.heapify(groups)

    partitions = []
    for seqlen, i in sorted_seqlen:
        current_group = heapq.heappop(groups)

        if current_group[0] + seqlen > max_token_len:
            partitions.append(current_group[3])
            new_group = [seqlen, 1, group_num, [i]]
            group_num = group_num + 1
            heapq.heappush(groups, new_group)
        else:
            # 将元素加入该组
            current_group[0] += seqlen  # 当前组总和增加
            current_group[1] += 1  # 当前组元素数量加1
            current_group[3].append(i)  # 当前组加入元素
            # 如果未满员，重新放回堆中
            heapq.heappush(groups, current_group)
    partitions.extend([group[3] for group in groups])
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool, max_token_len=None):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                        Requires len(seqlen_list) to be divisible by k_partitions.
                        If False, partitions can have varying numbers of items, focusing
                        only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                        original indices of the items assigned to that partition. The indices
                        within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    if len(seqlen_list) < k_partitions:
        raise AssertionError(f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]")

    def _check_and_sort_partitions(partitions):
        if len(partitions) != k_partitions:
            raise AssertionError(f"{len(partitions)} != {k_partitions}")
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            if len(partition) <= 0:
                raise AssertionError(f"the {i}-th partition is empty")
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        if seen_idx != set(range(len(seqlen_list))):
            raise AssertionError("seen_idx != set(range(len(seqlen_list)))")
        return sorted_partitions

    if dist.is_initialized() and max_token_len is not None:
        partitions = heapq_partition_with_max(seqlen_list=seqlen_list, k_partitions=k_partitions, max_token_len=max_token_len)
        k_partitions = torch.tensor(len(partitions))
        dist.all_reduce(k_partitions, op=dist.ReduceOp.MAX, group=None)
    partitions = heapq_partition(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)
    

class SeqlenBalancingPatch(NPUPatchHelper[seqlen_balancing]):
    def rearrange_micro_batches(
        batch,
        max_token_len,
        dp_group=None,
        num_batches_divided_by=None,
        same_micro_num_in_dp=True,
        min_num_micro_batch=None,
        use_dynamic_bsz_balance=True,
    ):
        """
        Split a batch into micro-batches by total token count, with optional DP sync and padding.

        Args:
            batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
            max_token_len (int): max sum of attention_mask per micro-batch.
            dp_group (optional): torch.distributed group for data-parallel sync.
            num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
            same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
            min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
            use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches

        Returns:
            List[TensorDict]: the micro-batches.
            List[List[int]]: index lists mapping each micro-batch back to original positions.
        """
        # this is per local micro_bsz
        max_seq_len = batch["attention_mask"].shape[-1]
        if max_token_len < max_seq_len:
            raise AssertionError(
            f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}")
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
        total_seqlen = seq_len_effective.sum().item()
        # NOTE: num_microbatches <= batch_size, so take the min of this two.
        num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
        if min_num_micro_batch is not None:
            # used to support pp
            num_micro_batches = max(min_num_micro_batch, num_micro_batches)
        if dist.is_initialized() and same_micro_num_in_dp:
            num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
            dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
            num_micro_batches = num_micro_batches.cpu().item()
        if num_batches_divided_by is not None:
            num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

        seq_len_effective = seq_len_effective.tolist()
        if num_micro_batches > len(seq_len_effective):
            raise AssertionError("num_micro_batches is larger than len(seq_len_effective)")
        micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False, max_token_len=max_token_len)

        if use_dynamic_bsz_balance:
            # Use the sum of squared sequence lengths to approximate attention computation workload
            micro_bsz_idx.sort(
                key=lambda partition: (
                    sum(seq_len_effective[idx] ** 2 for idx in partition),
                    min(partition) if partition else 0,
                ),
                reverse=True,
            )

        micro_batches = []

        for partition in micro_bsz_idx:
            curr_micro_batch = []
            for idx in partition:
                curr_micro_batch.append(batch[idx:idx + 1])
            curr_micro_batch = torch.cat(curr_micro_batch)

            micro_batches.append(curr_micro_batch)

        return micro_batches, micro_bsz_idx

