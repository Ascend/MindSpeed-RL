# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

"""Model and data parallel groups."""
import os
import re
import socket
import subprocess
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
import vllm.distributed.parallel_state as ps
import vllm_ascend.distributed.parallel_state as ascend_ps
import vllm.envs as envs
from vllm.config import get_current_vllm_config

from vllm.distributed.parallel_state import (
    get_pp_group,
    get_world_group,
    init_distributed_environment,
    init_model_parallel_group,
)

from mindspeed_rl.utils.loggers import Loggers

logger = Loggers(__name__)


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
# Expert model parallel group that the current rank belongs to.
_EP = None
# Expert tensor model parallel group that the current rank belongs to.
_ETP = None
# Data model parallel group that the current rank belongs to.
_DP = None

# Tensor model parallel group
_TP_GROUP_RANKS = None


def get_vllm_tp_group_ranks():
    return _TP_GROUP_RANKS


# This method is for initializing the ParallelGroup when using HybridEngine
def initialize_parallel_state(
        distributed_init_method: str = "env://",
        backend: str = "hccl",
        infer_tensor_model_parallel_size: int = 1,
        train_tensor_model_parallel_size: int = 1,
        infer_pipeline_model_parallel_size: int = 1,
        train_pipeline_model_parallel_size: int = 1,
        infer_expert_tensor_parallel_size: int = 1,
        train_expert_tensor_parallel_size: int = 1,
        train_expert_model_parallel_size: int = 1,
        infer_expert_model_parallel_size: int = 1,
        train_context_model_parallel_size: int = 1,
):
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Use the world_size set by TORCHRUN
    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    if world_size == -1:
        raise ValueError("The world_size is set to -1, not initialized by TORCHRUN")
    config = get_current_vllm_config()
    config.parallel_config.tensor_parallel_size = infer_tensor_model_parallel_size
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    if torch.distributed.get_world_size() > 1:
        # NOTE: build a sepearate inference group with infer tp & micro dp
        initialize_model_parallel_for_vllm(
            infer_tensor_model_parallel_size=infer_tensor_model_parallel_size,
            train_tensor_model_parallel_size=train_tensor_model_parallel_size,
            infer_pipeline_model_parallel_size=infer_pipeline_model_parallel_size,
            train_pipeline_model_parallel_size=train_pipeline_model_parallel_size,
            infer_expert_tensor_parallel_size=infer_expert_tensor_parallel_size,
            train_expert_tensor_parallel_size=train_expert_tensor_parallel_size,
            train_expert_model_parallel_size=train_expert_model_parallel_size,
            infer_expert_model_parallel_size=infer_expert_model_parallel_size,
            train_context_model_parallel_size=train_context_model_parallel_size
        )
    else:
        initialize_model_parallel(infer_tensor_model_parallel_size, infer_pipeline_model_parallel_size, backend)


def initialize_model_parallel_for_vllm(
        infer_tensor_model_parallel_size: int,
        train_tensor_model_parallel_size: int = 1,
        infer_pipeline_model_parallel_size: int = 1,
        train_pipeline_model_parallel_size: int = 1,
        infer_expert_tensor_parallel_size: int = 1,
        train_expert_tensor_parallel_size: int = 1,
        train_expert_model_parallel_size: int = 1,
        infer_expert_model_parallel_size: int = 1,
        train_context_model_parallel_size: int = 1,
        num_process: int = 1,
        rebulid_EP_group: bool = False
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
        if ((world_size // (
                train_tensor_model_parallel_size * train_pipeline_model_parallel_size)) * train_tensor_model_parallel_size < infer_tensor_model_parallel_size or
                ((world_size // (
                        train_tensor_model_parallel_size * train_pipeline_model_parallel_size)) * train_tensor_model_parallel_size) % infer_tensor_model_parallel_size != 0):
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
            raise ValueError(
                f"Can't gather train tp size {train_tensor_model_parallel_size} to infer tp size {infer_tensor_model_parallel_size}")
        num_tensor_model_parallel_groups = world_size // infer_tensor_model_parallel_size
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = list(range(i * infer_tensor_model_parallel_size, (i + 1) * infer_tensor_model_parallel_size))
            group_ranks.append(ranks)

        return group_ranks

    def get_tp_group_ranks():
        if infer_tensor_model_parallel_size > train_tensor_model_parallel_size:
            tp_group_ranks = get_split_tp_group_ranks()
        else:
            tp_group_ranks = get_allgather_tp_group_ranks()
        global _TP_GROUP_RANKS
        _TP_GROUP_RANKS = tp_group_ranks
        return tp_group_ranks

    tp_group_ranks = get_tp_group_ranks()
    logger.info(f"TP rank: {tp_group_ranks}")

    _TP = init_model_parallel_group(
        group_ranks=tp_group_ranks,
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
    logger.info(f"PP rank: {group_ranks}")
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend,
    )
    ps._PP = _PP  # for verl

    data_parallel_size = 1
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()

    if config is not None:
        data_parallel_size = config.parallel_config.data_parallel_size

    num_expert_parallel_groups: int = infer_expert_tensor_parallel_size
    num_expert_tensor_parallel_groups: int = world_size // infer_expert_tensor_parallel_size

    num_rank_per_process = world_size // num_process
    all_ranks = list(range(world_size))

    global _EP
    assert _EP is None, ("expert parallel group is already initialized")
    group_ranks = []

    if rebulid_EP_group:
        # 重新建组
        tensor_model_parallel_size = train_tensor_model_parallel_size
        context_parallel_size = train_context_model_parallel_size
        expert_model_parallel_size = train_expert_model_parallel_size
        train_data_parallel_size = world_size // tensor_model_parallel_size // train_pipeline_model_parallel_size 
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * train_data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        num_expert_groups: int = train_data_parallel_size * context_parallel_size // expert_model_parallel_size
        tensor_and_expert_group_size = tensor_model_parallel_size * expert_model_parallel_size
        all_tensor_and_expert_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size_with_cp + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                all_tensor_and_expert_group_ranks.append(list(ranks))
        train_all_tensor_and_expert_group_ranks_tensor = torch.tensor(all_tensor_and_expert_group_ranks)
        # 将训练态的EPG按照推理EP进行转置
        infer_actual_expert_model_parallel_size = infer_tensor_model_parallel_size * infer_expert_model_parallel_size
        experts_memory_expend_N = infer_actual_expert_model_parallel_size // tensor_and_expert_group_size
        ep_group_num = world_size // tensor_and_expert_group_size
        group_ranks = []
        for i in range(0, ep_group_num, experts_memory_expend_N):
            per_ep_group = train_all_tensor_and_expert_group_ranks_tensor[i:i + experts_memory_expend_N]
            per_ep_group_T = per_ep_group.T
            ranks = per_ep_group_T.reshape(-1).tolist()
            group_ranks.append(ranks)
        logger.info(f"EP rank: {group_ranks}")

    else:
        # 保序
        group_ranks = []
        tensor_model_parallel_size = infer_tensor_model_parallel_size
        context_parallel_size = 1
        expert_model_parallel_size = infer_expert_model_parallel_size
        infer_data_parallel_size = world_size // tensor_model_parallel_size // infer_pipeline_model_parallel_size 
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * infer_data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        num_expert_groups: int = infer_data_parallel_size * context_parallel_size // expert_model_parallel_size
        tensor_and_expert_group_size = tensor_model_parallel_size * expert_model_parallel_size
        group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size_with_cp + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                group_ranks.append(list(ranks))
        logger.info(f"EP rank: {group_ranks}")

    ascend_ps._EP = init_model_parallel_group(group_ranks,
                                       get_world_group().local_rank,
                                       backend,
                                       group_name="ep")

    global _ETP
    assert _ETP is None, (
        "expert tensor parallel group is already initialized")

    group_ranks = []
    for i in range(num_expert_tensor_parallel_groups):
        ranks = list(range(i * infer_expert_tensor_parallel_size,
                           (i + 1) * infer_expert_tensor_parallel_size))
        group_ranks.append(ranks)
    logger.info(f"ETP rank: {group_ranks}")

    ascend_ps._ETP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="etp")
    
    if data_parallel_size > 1:
        global _DP
        assert _DP is None, ("data parallel group is already initialized")
        dp_group_ranks = torch.tensor(tp_group_ranks).transpose(0, 1).reshape(-1, data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in dp_group_ranks]
        logger.info(f"DP rank: {group_ranks}")

        ps._DP = init_model_parallel_group(group_ranks,
                                           get_world_group().local_rank,
                                           backend,
                                           group_name="dp")

        os.environ["VLLM_DP_RANK"] = str(ps._DP.rank_in_group)
        envs.VLLM_DP_RANK = int(os.environ["VLLM_DP_RANK"])
        ip_list = get_cluster_info()

        for index, group_rank in enumerate(group_ranks):
            if torch.distributed.get_rank() in group_rank:
                os.environ["VLLM_DP_MASTER_PORT"] = str(
                    int(os.environ.get("MASTER_PORT")) + 1 + index)
                os.environ["VLLM_DP_MASTER_IP"] = ip_list[group_rank[0]]

        envs.VLLM_DP_MASTER_IP = os.environ["VLLM_DP_MASTER_IP"]
        envs.VLLM_DP_MASTER_PORT = int(os.environ["VLLM_DP_MASTER_PORT"])
        os.environ["VLLM_PORT"] = os.environ["VLLM_DP_MASTER_PORT"]
        envs.VLLM_PORT = envs.VLLM_DP_MASTER_PORT

        logger.info(f"rank: {torch.distributed.get_rank()}>>>>>>VLLM_DP_MASTER_IP: {envs.VLLM_DP_MASTER_IP}, VLLM_DP_MASTER_PORT: {envs.VLLM_DP_MASTER_PORT}")


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


def get_cluster_info():
    # 确保分布式环境已初始化
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized")

    world_size = dist.get_world_size()

    # 获取当前节点的IP地址
    ip_address = _get_current_node_ip()

    # 收集所有rank的IP地址
    ip_list = [None] * world_size
    dist.all_gather_object(ip_list, ip_address)

    return ip_list


def _get_current_node_ip() -> str:
    try:
        # 创建一个 UDP 套接字（仅用于获取接口信息）
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 连接到一个外部地址（无需真实通信）
            s.connect(("8.8.8.8", 80))  # Google DNS 服务器
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = _get_ip_by_ifname()
        if not local_ip:
            # 如果失败，回退到遍历接口
            local_ip = "127.0.0.1"
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                ip = addr[4][0]
                if not ip.startswith("::"):
                    local_ip = ip
                    break
    return local_ip


def _get_ip_by_ifname():
    """
    通过接口名称（如 eth0、en0）获取 IPv4 地址
    返回 IP 字符串，失败返回 None
    """
    try:
        # 执行 ifconfig 命令并捕获输出
        ifname = os.environ.get("HCCL_SOCKET_IFNAME", 0)
        if ifname:
            output = subprocess.check_output(["ifconfig", ifname], stderr=subprocess.STDOUT).decode()
            # 正则匹配 IPv4 地址（排除 127.0.0.1）
            matches = re.findall(r'inet (?:addr:)?((?:\d{1,3}\.){3}\d{1,3})', output)
            for ip in matches:
                if ip != "127.0.0.1":
                    return ip
        return None
    except subprocess.CalledProcessError:
        return None
