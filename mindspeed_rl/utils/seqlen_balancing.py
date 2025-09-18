# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import copy
import heapq
from typing import List, Tuple

import torch
import torch.distributed as dist

from mindspeed_rl.utils import is_multimodal


def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """Karmarkar-Karp algorithm for partitioning a list of integers into k partitions
    such that the difference between the largest and smallest partition is minimized.
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            number of partitions
        equal_size (bool):
            whether to make partitions equal size
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for s in self.sets:
                cur_partition = []
                for idx, _ in s.items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        if len(seqlen_list) % k_partitions != 0:
            raise ValueError(f"{len(seqlen_list)} % {k_partitions} != 0")
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for partition in partitions:
            if len(partition) * k_partitions != len(seqlen_list):
                raise ValueError(f"{len(partition)} * {k_partitions} != {len(seqlen_list)}")
    return partitions


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


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seq length across dp ranks and micro batches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    if k_partitions > len(seqlen_list):
        raise ValueError(f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]")

    def _check_and_sort_partitions(partitions):
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        return sorted_partitions

    partitions = heapq_partition(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def balanced_bin_packing(seqlen_list: List[int], max_capacity: int):
    """Balanced bin packing algorithm that ensures each bin doesn't exceed max_capacity
    while maintaining load balance across bins.

    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        max_capacity (int):
            maximum capacity for each bin/partition

    Returns:
        partitions (List[List[int]]):
            list of partitions, each containing indices of items
    """
    if not seqlen_list:
        return []

    # Create list of (seqlen, original_index) and sort by seqlen descending
    indexed_seqlens = [(seqlen, i) for i, seqlen in enumerate(seqlen_list)]
    indexed_seqlens.sort(reverse=True)  # Largest first (Best Fit Decreasing)

    # Initialize bins with their current capacity usage
    bins = []  # Each bin: {'items': [(idx, seqlen), ...], 'capacity_used': int}

    for seqlen, original_idx in indexed_seqlens:
        if seqlen > max_capacity:
            raise ValueError(f"Item with seqlen {seqlen} exceeds max_capacity {max_capacity}")

        # Find the best bin that can accommodate this item
        best_bin_idx = -1
        best_remaining_capacity = max_capacity + 1  # Initialize to impossible value

        for bin_idx, bin_info in enumerate(bins):
            remaining_capacity = max_capacity - bin_info['capacity_used']
            # Check if item fits and this bin has less remaining capacity (Best Fit)
            if remaining_capacity >= seqlen and remaining_capacity < best_remaining_capacity:
                best_bin_idx = bin_idx
                best_remaining_capacity = remaining_capacity

        if best_bin_idx != -1:
            # Add to existing bin
            bins[best_bin_idx]['items'].append((original_idx, seqlen))
            bins[best_bin_idx]['capacity_used'] += seqlen
        else:
            # Create new bin
            bins.append({
                'items': [(original_idx, seqlen)],
                'capacity_used': seqlen
            })

    # Post-processing: Try to balance the bins by moving items between them
    # This helps reduce the variance in bin loads
    _balance_bins(bins, max_capacity)

    # Convert to partition format (list of indices for each partition)
    partitions = []
    for bin_info in bins:
        partition = [idx for idx, _ in bin_info['items']]
        partitions.append(partition)

    return partitions


def _balance_bins(bins: List[dict], max_capacity: int):
    """Helper function to balance loads across bins by moving items between bins.

    Parameters:
        bins: List of bin dictionaries with 'items' and 'capacity_used' keys
        max_capacity: Maximum capacity per bin
    """
    if len(bins) <= 1:
        return

    # Perform multiple passes to improve balance
    max_iterations = 3
    for _ in range(max_iterations):
        improved = False

        # Sort bins by current load
        bins.sort(key=lambda b: b['capacity_used'])

        # Try to move items from heaviest bins to lightest bins
        for heavy_idx in range(len(bins) - 1, 0, -1):
            heavy_bin = bins[heavy_idx]

            for light_idx in range(heavy_idx):
                light_bin = bins[light_idx]

                # Calculate load difference
                load_diff = heavy_bin['capacity_used'] - light_bin['capacity_used']
                if load_diff <= 1:  # Already balanced enough
                    break

                # Find items in heavy bin that can be moved to light bin
                for item_idx, (idx, seqlen) in enumerate(heavy_bin['items']):
                    new_light_load = light_bin['capacity_used'] + seqlen
                    new_heavy_load = heavy_bin['capacity_used'] - seqlen

                    # Check if move is beneficial and doesn't violate capacity
                    if (new_light_load <= max_capacity and
                            abs(new_heavy_load - new_light_load) < load_diff):
                        # Move the item
                        item = heavy_bin['items'].pop(item_idx)
                        light_bin['items'].append(item)
                        heavy_bin['capacity_used'] -= seqlen
                        light_bin['capacity_used'] += seqlen
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break


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


def rearrange_micro_batches(
    seqlen_list: List[int],
    max_token_len: int,
    dynamic_max_batch_size: int = None,
    dp_group=None
):
    """
    Rearranges micro batches to balance sequence lengths across partitions,
    ensuring no partition exceeds max_token_len.

    Parameters:
        seqlen_list (List[int]): Sequence lengths of each item.
        max_token_len (int): Maximum allowed sum of sequence lengths per partition.
        dynamic_max_batch_size (int, optional): Minimum number of partitions based on batch size.
        dp_group: Distributed process group for synchronization (optional).

    Returns:
        List[List[int]]: List of partitions, each containing indices of items.
    """
    if is_multimodal():
        # When multimodal, max_token_len is a list representing the maximum token length for each modality.
        # Use balanced bin packing algorithm with capacity constraints
        return balanced_bin_packing(seqlen_list=seqlen_list, max_capacity=max_token_len)

    # Calculate the minimum number of bins
    total_sum_of_seqlen = sum(seqlen_list)
    k_partitions = (total_sum_of_seqlen + max_token_len - 1) // max_token_len

    partitions = heapq_partition_with_max(seqlen_list=seqlen_list, k_partitions=k_partitions, max_token_len=max_token_len)
    k_partitions = torch.tensor(len(partitions))

    if dynamic_max_batch_size is not None:
        k_partitions = max(k_partitions, (len(seqlen_list) + dynamic_max_batch_size - 1) // dynamic_max_batch_size)
        
    if dist.is_initialized():
        k_partitions = torch.tensor([k_partitions], device='npu')
        dist.all_reduce(k_partitions, op=dist.ReduceOp.MAX, group=dp_group)
        k_partitions = k_partitions.cpu().item()

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=False)

    return partitions


def get_reverse_idx(idx_map):
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map