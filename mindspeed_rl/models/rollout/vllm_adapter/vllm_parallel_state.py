# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

"""Model and data parallel groups."""
import os
from typing import Optional

import torch
import torch.distributed
import vllm.distributed.parallel_state as ps

from vllm.distributed.parallel_state import (
    get_pp_group,
    get_world_group,
    init_distributed_environment,
    init_model_parallel_group,
)


"""
This version is strongly tied with Megatron to implement HybridEngine and weight sharing between vllm and Megatron.
- We assume the Megatron tp+dp+pp world is already established before calling this function.

"""

# Device mesh for using DTensor
_DEVICE_MESH = None

# Tensor model parallel group that the current rank belongs to.
_TP = None
# Pipeline model parallel group that the current rank belongs to.
_PP = None


# This method is for initializing the ParallelGroup when using HybridEngine
def initialize_parallel_state(
    distributed_init_method: str = "env://",
    backend: str = "hccl",
    infer_tensor_model_parallel_size: int = 1,
    train_tensor_model_parallel_size: int = 1,
    infer_pipeline_model_parallel_size: int = 1,
    train_pipeline_model_parallel_size: int = 1
):
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"


    # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Use the world_size set by TORCHRUN
    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    if world_size == -1:
        raise ValueError("The world_size is set to -1, not initialized by TORCHRUN")
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    if torch.distributed.get_world_size() > 1:
        # NOTE: build a sepearate inference group with infer tp & micro dp
        initialize_model_parallel_for_vllm(
            infer_tensor_model_parallel_size=infer_tensor_model_parallel_size,
            train_tensor_model_parallel_size=train_tensor_model_parallel_size,
            infer_pipeline_model_parallel_size=infer_pipeline_model_parallel_size,
            train_pipeline_model_parallel_size=train_pipeline_model_parallel_size
        )
    else:
        initialize_model_parallel(infer_tensor_model_parallel_size, infer_pipeline_model_parallel_size, backend)


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)
        return

    current_tp_size = get_tensor_model_parallel_world_size()
    if current_tp_size != tensor_model_parallel_size:
        raise ValueError(
            "tensor parallel group already initialized, but of unexpected size: "
            f"{current_tp_size=} vs. "
            f"{tensor_model_parallel_size=}"
        )
    pp_world_size = get_pp_group().world_size
    if pp_world_size != pipeline_model_parallel_size:
        raise ValueError(
            "pipeline parallel group already initialized, but of unexpected size: "
            f"{pp_world_size=} vs. "
            f"{pipeline_model_parallel_size=}"
        )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return ps._TP is not None
    # and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


def initialize_model_parallel_for_vllm(
    infer_tensor_model_parallel_size: int,
    train_tensor_model_parallel_size: int = 1,
    infer_pipeline_model_parallel_size: int = 1,
    train_pipeline_model_parallel_size: int = 1
) -> None:

    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise ValueError("torch.distributed is not initialized")

    if not isinstance(infer_tensor_model_parallel_size, int):
        raise TypeError("tensor_model_parallel_size must be an integer")

    # Build the tensor model-parallel groups.
    if ps._TP is not None:
        raise ValueError("tensor model parallel group is already initialized")

    global _TP

    world_size: int = torch.distributed.get_world_size()

    backend = torch.distributed.get_backend()

    def get_split_tp_group_ranks():
        '''
        Arguments:
            infer_tensor_model_parallel_size: number of GPUs used for infer tensor model
                parallelism.

        Each group_ranks is in order of tp ascending.

        Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
        use 2 GPUs to parallelize the model tensor. The present function will
        create 4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        Returns: list of group_lists
            [[g0, g1], [g2, g3], [g4, g5], [g6, g7]]
        '''
        if ((world_size // (train_tensor_model_parallel_size * train_pipeline_model_parallel_size)) * train_tensor_model_parallel_size < infer_tensor_model_parallel_size or
                ((world_size // (train_tensor_model_parallel_size * train_pipeline_model_parallel_size)) * train_tensor_model_parallel_size) % infer_tensor_model_parallel_size != 0):
            raise ValueError(
                f"Can't split train tp size {train_tensor_model_parallel_size} to infer tp size {infer_tensor_model_parallel_size} "
                f"with train dp size {(world_size // (train_tensor_model_parallel_size * train_pipeline_model_parallel_size))}.")
        group_ranks = []
        for i in range(world_size // infer_tensor_model_parallel_size):
            ranks = list(range(i * infer_tensor_model_parallel_size, (i + 1) * infer_tensor_model_parallel_size))
            group_ranks.append(ranks)
        return group_ranks

    def get_allgather_tp_group_ranks():
        '''
        Arguments:
            train_tensor_model_parallel_size: number of GPUs used for train tensor model
                parallelism.
            infer_tensor_model_parallel_size: number of GPUs used for infer tensor model
                parallelism.

        Each group_ranks is in order of tp ascending.

        Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
        use 4 GPUs to parallelize the model tensor for train, 2 GPUs to parallelize the
        model tensor for infer with 2 data parallel groups. The present function will
        create 4 tensor model-parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7]
        Returns: list of group_lists
            [[g0, g2], [g1, g3], [g4, g6], [g5, g7]]
        '''
        if train_tensor_model_parallel_size < infer_tensor_model_parallel_size or train_tensor_model_parallel_size % infer_tensor_model_parallel_size != 0:
            raise ValueError(f"Can't gather train tp size {train_tensor_model_parallel_size} to infer tp size {infer_tensor_model_parallel_size}")
        num_tensor_model_parallel_groups = world_size // infer_tensor_model_parallel_size
        num_tensor_model_parallel_groups_per_train_tp = train_tensor_model_parallel_size // infer_tensor_model_parallel_size
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            start = train_tensor_model_parallel_size * i
            end = train_tensor_model_parallel_size * (i + 1)
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                ranks = list(range(start + j, end, num_tensor_model_parallel_groups_per_train_tp))
                group_ranks.append(ranks)

        return group_ranks

    def get_tp_group_ranks():
        if infer_tensor_model_parallel_size > train_tensor_model_parallel_size:
            return get_split_tp_group_ranks()
        else:
            return get_allgather_tp_group_ranks()


    _TP = init_model_parallel_group(
        group_ranks=get_tp_group_ranks(),
        local_rank=get_world_group().local_rank,
        backend=backend,
        use_message_queue_broadcaster=True,
    )
    ps._TP = _TP
    num_pipeline_model_parallel_groups: int = world_size // infer_pipeline_model_parallel_size
    global _PP
    if _PP is not None:
        raise ValueError("pipeline model parallel group is already initialized")
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend,
    )
    ps._PP = _PP  # for verl


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    NOTE: This method is a hack from the open-sourced version without
    asertion of world_size = tp * pp

    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise ValueError("torch.distributed is not initialized")
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(ps.get_world_group().device_group)



    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    global _TP
    if _TP is not None:
        raise ValueError("tensor model parallel group is already initialized")
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
    )
    ps._TP = _TP
    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP
    if _PP is not None:
        raise ValueError("pipeline model parallel group is already initialized")
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend,
    )

    ps._PP = _PP  # for verl

