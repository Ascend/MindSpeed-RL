# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import copy
import heapq
from typing import List, Tuple

import torch
import torch.distributed as dist


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


def rearrange_micro_batches(seqlen_list: List[int], max_token_len: int, dp_group=None):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seq length across dp ranks and micro batches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        max_token_len (int):
     Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    if max(seqlen_list) > max_token_len:
        raise ValueError(f"seqlen of items:[{max(seqlen_list)}] must <= max_token_len:[{max_token_len}]")
    

    # Calculate the minimum number of bins
    total_sum_of_seqlen = sum(seqlen_list)
    if total_sum_of_seqlen % max_token_len == 0:
        k_partitions = total_sum_of_seqlen // max_token_len
    else:
        k_partitions = total_sum_of_seqlen // max_token_len + 1

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