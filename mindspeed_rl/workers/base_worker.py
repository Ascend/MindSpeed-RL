# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import os
from abc import ABC
from typing import Any, Callable, List
import socket
from dataclasses import dataclass

import torch
import torch_npu
import ray

from mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state import get_vllm_tp_group_ranks
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.pad_process import remove_padding_tensor_dict_to_dict, padding_dict_to_tensor_dict
from mindspeed_rl.utils.tokenizer import BaseTokenizer

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.trainer.utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_src_rank,
    get_pipeline_model_parallel_group,
    is_pipeline_last_stage,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_context_parallel_rank,
    get_tensor_model_parallel_src_rank,
    get_model_parallel_group,
    get_context_parallel_src_rank,
    get_context_parallel_group
)
from mindspeed_rl.utils.compute import set_parallel_state, set_vocab_parallel
from mindspeed_rl.utils.utils import get_current_dp_range_indexes, get_current_node_ip
from mindspeed_rl.trainer.utils.transfer_dock import pack_experience_columns, unpack_pad_experience
from mindspeed_rl.trainer.utils.data_strategy import unpack_mm_experience
from mindspeed_rl.utils.utils import mstx_timer_decorator, is_multimodal
from mindspeed_rl.utils.zmq_communication import (ZmqServer, ZmqClient, ZmqServerInfo, ZmqClientInfo,
                                                  ZMQ_ROLE_NONE, ZMQ_ROLE_CLIENT, ZMQ_ROLE_SERVER)

logger = Loggers("base_worker")

_DP_RANGE_DATA_CONSUMED_FLAG = 0

_DP_RANGE_DATA_NOT_CONSUMED_FLAG = 1


@dataclass
class BroadcastHcclParameter:
    rank_flg: bool
    experience_columns: List[str]
    batch_data: Any
    batch_data_length: int
    use_vllm: bool
    index: Any


@dataclass
class DispatchZmqParameter:
    experience_consumer_stage: str
    experience_columns: List[str]
    experience_count: int
    tp_size: int = 1
    cp_size: int = 1
    cp_algo: str = None
    use_vllm: bool = False
    indexes: List[int] = None
    get_n_samples: bool = True
    enable_partial_rollout: bool = False
    is_generate: bool = False


class BaseRayWorker:
    def __init__(self):
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        self._local_rank = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
        torch.npu.set_device(self._local_rank)
        current_device = torch.npu.current_device()
        if os.environ.get("MASTER_ADDR", 0) == "localhost":
            self._master_addr = get_current_node_ip()
            self._master_port = self._get_free_port()
            os.environ["MASTER_ADDR"] = self._master_addr
            os.environ["MASTER_PORT"] = str(self._master_port)
        else:
            self._master_addr = os.environ.get("MASTER_ADDR")
            self._master_port = os.environ.get("MASTER_PORT")
        os.environ["LOCAL_RANK"] = str(self._local_rank)
        self.zmq_role = ZMQ_ROLE_NONE
        logger.info(f"worker init begin, current device id: {current_device}, rank: {self._rank},"
                    f" world_size: {self._world_size}, local rank: {self._local_rank},"
                    f" master_addr: {self._master_addr}, master_port: {self._master_port}")

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind((get_current_node_ip(), 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BaseWorker(BaseRayWorker, ABC):
    """基类，封装通用逻辑但保留子类接口和装饰器"""

    def __init__(
            self,
            megatron_config: MegatronConfig = None,
            rl_config: RLConfig = None,
            generate_config: GenerateConfig = None,
            model_provider: Callable = None,
            initialize_func: Callable = None,
            get_megatron_module: Callable = None,
            tokenizer: BaseTokenizer = None,
            profiler_config: ProfilerConfig = None,
            msprobe_config: MsprobeConfig = None,
            **kwargs
    ):
        super().__init__()
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        self.rl_config = rl_config
        self.megatron_config = megatron_config
        self.generate_config = generate_config
        self.profiler_config = profiler_config
        self.msprobe_config = msprobe_config
        self.model_provider = model_provider
        self.initialize_func = initialize_func
        self.get_megatron_module = get_megatron_module
        self.tokenizer = tokenizer
        self.megatron_config.update(kwargs)
        self.inference_model = None
        self.sharding_manager = None
        self.hybrid_engine = None
        self.opt_param_scheduler = None
        self.optimizer = None
        self.model_type = None
        self.model = None
        self.td = None
        self.mm_td = None
        self.sampling_transfer_dock = None
        self.args = None
        self.zmq_server = None
        self.zmq_client = None
        self.zmq_role = ZMQ_ROLE_NONE

    @mstx_timer_decorator
    def all_consumed(self, experience_consumer_stage, sorted_indexes, use_vllm=False, is_generate=False):
        if self.rl_config.guarantee_order and not sorted_indexes:
            return _DP_RANGE_DATA_CONSUMED_FLAG
        elif self.rl_config.guarantee_order:
            return _DP_RANGE_DATA_NOT_CONSUMED_FLAG
        if use_vllm:
            current_device = next(self.inference_model.model.parameters()).device
        else:
            current_device = next(self.model[0].parameters()).device
        status = torch.tensor(0, device=current_device)

        if not use_vllm:
            rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_context_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)
        else:
            rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)
        if rank_flg:
            get_n_samples = not (getattr(self.rl_config, "partial_rollout_max_split", 1) > 1 and is_generate)
            if self.sampling_transfer_dock and is_generate:
                status = torch.tensor(
                    int(not ray.get(self.sampling_transfer_dock.all_consumed.remote(experience_consumer_stage, get_n_samples))),
                    device=current_device,
                )
            else:
                status = torch.tensor(
                    int(not ray.get(self.td.all_consumed.remote(experience_consumer_stage, get_n_samples))),
                    device=current_device,
                )
        torch.distributed.all_reduce(status, group=get_model_parallel_group(self.parallel_state, use_vllm),
                                     op=torch.distributed.ReduceOp.MAX)
        if not use_vllm:
            torch.distributed.all_reduce(status, group=get_context_parallel_group(self.parallel_state, use_vllm),
                                     op=torch.distributed.ReduceOp.MAX)

        return status

    def setup_distributed_rank(self):
        logger.info(f"getenv RANK         : {os.getenv('RANK')}")
        logger.info(f"getenv WORLD_SIZE   : {os.getenv('WORLD_SIZE')}")
        logger.info(f"getenv LOCAL_RANK   : {os.getenv('LOCAL_RANK')}")
        logger.info(f"getenv MASTER_ADDR  : {os.getenv('MASTER_ADDR')}")
        logger.info(f"getenv MASTER_PORT  : {os.getenv('MASTER_PORT')}")
        logger.info(f"ray alloc NPU ID    :  {int(ray.get_runtime_context().get_accelerator_ids()['NPU'][0])}")

        import copy
        config = copy.deepcopy(self.megatron_config)
        if config.stage == "ray_dapo":
            config.stage = "ray_grpo"

        self.initialize_func(config=config)
        megatron_module = self.get_megatron_module()
        for key, value in megatron_module.items():
            setattr(self, key, value)

        set_parallel_state(self.parallel_state)
        set_vocab_parallel(self.vocab_parallel_cross_entropy)
        self.args = self.get_args()
        self.forward_backward_func = self.get_forward_backward_func()

        if self.rl_config.zmq_communication:
            if (get_tensor_model_parallel_rank(self.parallel_state, False) == 0 and
                get_context_parallel_rank(self.parallel_state, False) == 0 and
                get_pipeline_model_parallel_rank(self.parallel_state, False) == 0):
                server_info = ZmqServerInfo()
                server_info.global_rank = self._rank
                server_info.dp_world_size = (self.parallel_state.get_tensor_model_parallel_world_size() *
                                             self.parallel_state.get_pipeline_model_parallel_world_size() *
                                             self.parallel_state.get_context_parallel_world_size())
                server_info.ip_addr = ray._private.services.get_node_ip_address().strip("[]")
                server_info.register_port = self._get_free_port()
                server_info.publisher_port = self._get_free_port()
                server_info.reliability_port = self._get_free_port()
                self.zmq_role = ZMQ_ROLE_SERVER
                self.zmq_server = ZmqServer(server_info, self.parallel_state)
            else:
                client_info = ZmqClientInfo()
                client_info.global_rank = self._rank
                self.zmq_role = ZMQ_ROLE_CLIENT
                self.zmq_client = ZmqClient(client_info, self.parallel_state)

    def initialize(self):
        """
        Initialize models. These models perform actual training and inference operations. For details,
        see BaseTrainingEngine and BaseInferEngine.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def init_transfer_dock(self, td, sampling_transfer_dock=None):
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def td(self):
        """
        worker需要设置td（数据队列）后才可以使用，这里添加判断
        """
        if self._td is None:
            raise ValueError("Transfer Dock is not initialized")
        return self._td

    @td.setter
    def td(self, value):
        self._td = value

    @property
    def mm_td(self):
        """
        worker需要设置td（数据队列）后才可以使用，这里添加判断
        """
        if self._mm_td is None:
            raise ValueError("MultiModal Transfer Dock is not initialized")
        return self._mm_td

    @mm_td.setter
    def mm_td(self, value):
        self._mm_td = value

    @mstx_timer_decorator
    def empty_cache(self):
        """Clear GPU cache (can be overridden by subclasses)"""
        torch.cuda.empty_cache()

    @staticmethod
    def compute_batch_data_size(batch_data):
        size = 0
        for data in batch_data.values():
            if data.is_sparse:
                # 稀疏张量的内存占用
                storage = data.storage()
                element_size = storage.element_size()
                num_elements = storage.numel()
                total_bytes = num_elements * element_size
            else:
                element_size = data.element_size()
                num_elements = data.numel()
                total_bytes = num_elements * element_size
            size += total_bytes
        return size

    def broadcast_data_hccl(self, data: BroadcastHcclParameter):
        rank_flg = data.rank_flg
        experience_columns = data.experience_columns
        batch_data = data.batch_data
        batch_data_length = data.batch_data_length
        use_vllm = data.use_vllm
        index = data.index

        index_without_pad = []
        for key in experience_columns:
            if rank_flg:
                batch_data_shape = torch.tensor(batch_data[key].shape,
                                                dtype=torch.int64, device=torch.cuda.current_device())

                batch_data_length_shape = torch.tensor(batch_data_length[key].shape, dtype=torch.int64,
                                                       device=torch.cuda.current_device())

                if batch_data[key].dtype == torch.int32:
                    batch_data_dtype = torch.tensor(1,
                                                    dtype=torch.int64, device=torch.cuda.current_device())
                else:
                    batch_data_dtype = torch.tensor(2,
                                                    dtype=torch.int64, device=torch.cuda.current_device())

                # 添加维度信息
                if key not in batch_data.keys():
                    raise KeyError(f'{key} is missing!')
                batch_data_ndim = torch.tensor(len(batch_data[key].shape),
                                               dtype=torch.int64, device=torch.cuda.current_device())
            else:
                batch_data_shape = torch.empty(2, device=torch.cuda.current_device(), dtype=torch.int64)  # 最多支持二维张量
                batch_data_dtype = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)
                batch_data_length_shape = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)
                batch_data_ndim = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)

            # TP domain sync
            torch.distributed.broadcast(
                batch_data_shape, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_dtype, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(batch_data_length_shape,
                                        get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                                        group=get_tensor_model_parallel_group(self.parallel_state, use_vllm))
            torch.distributed.broadcast(
                batch_data_ndim, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            # CP domain sync
            if not use_vllm:
                torch.distributed.broadcast(
                    batch_data_shape, get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )
                torch.distributed.broadcast(
                    batch_data_dtype, get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )
                torch.distributed.broadcast(batch_data_length_shape,
                                            get_context_parallel_src_rank(self.parallel_state, use_vllm),
                                            group=get_context_parallel_group(self.parallel_state, use_vllm))
                torch.distributed.broadcast(
                    batch_data_ndim, get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )
            # PP domain sync
            torch.distributed.broadcast(
                batch_data_shape, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_dtype, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(batch_data_length_shape,
                                        get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                                        group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm))
            torch.distributed.broadcast(
                batch_data_ndim, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )

            if not rank_flg:
                if batch_data_ndim == 1: # 一维张量处理
                    if batch_data_dtype == 1:
                        batch_data[key] = torch.empty(batch_data_shape[0],   # batch_data_shape[1],
                                                    device=torch.cuda.current_device(),
                                                    dtype=torch.int32)
                    else:
                        batch_data[key] = torch.empty(batch_data_shape[0],   # batch_data_shape[1],
                                                    device=torch.cuda.current_device(),
                                                    dtype=torch.float32)
                else: # 二维张量处理
                    if batch_data_dtype == 1:
                        batch_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                    device=torch.cuda.current_device(),
                                                    dtype=torch.int32)
                    else:
                        batch_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                    device=torch.cuda.current_device(),
                                                    dtype=torch.float32)
                batch_data_length[key] = torch.empty(batch_data_length_shape[0],
                                                     device=torch.cuda.current_device(), dtype=torch.int32)

            # 传输tensor数据
            torch.distributed.broadcast(
                batch_data[key].cuda(), get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )

            if not use_vllm:
                torch.distributed.broadcast(
                    batch_data[key].cuda(), get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )

            torch.distributed.broadcast(
                batch_data[key].cuda(), get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )

            torch.distributed.broadcast(batch_data_length[key].cuda(),
                                        get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                                        group=get_tensor_model_parallel_group(self.parallel_state, use_vllm))

            if not use_vllm:
                torch.distributed.broadcast(batch_data_length[key].cuda(),
                                            get_context_parallel_src_rank(self.parallel_state, use_vllm),
                                            group=get_context_parallel_group(self.parallel_state, use_vllm))

            torch.distributed.broadcast(batch_data_length[key].cuda(),
                                        get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                                        group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm))
            index_without_pad = index.cpu().numpy().tolist()[:batch_data_length_shape[0]]

        return index_without_pad

    def dispatch_transfer_dock_data_zmq(self, para: DispatchZmqParameter):
        experience_consumer_stage = para.experience_consumer_stage
        experience_columns = para.experience_columns
        experience_count = para.experience_count
        tp_size = para.tp_size
        cp_size = para.cp_size
        cp_algo = para.cp_algo
        use_vllm = para.use_vllm
        indexes = para.indexes
        get_n_samples = para.get_n_samples
        enable_partial_rollout = para.enable_partial_rollout
        is_generate = para.is_generate

        if use_vllm:
            server_flag = True if hasattr(self, "zmq_server_vllm") else False
            server = self.zmq_server_vllm if hasattr(self, "zmq_server_vllm") else None
            client = self.zmq_client_vllm if hasattr(self, "zmq_client_vllm") else None
        else:
            server_flag = True if self.zmq_server is not None else False
            server = self.zmq_server if self.zmq_server is not None else None
            client = self.zmq_client if self.zmq_client is not None else None

        if is_multimodal():
            if self.sampling_transfer_dock and is_generate:
                mm_columns = ray.get(self.mm_sampling_transfer_dock.get_columns.remote(experience_consumer_stage))
            else:
                mm_columns = ray.get(self.mm_td.get_columns.remote(experience_consumer_stage))
        else:
            mm_columns = []

        batch_data = {}
        batch_data_length = {}
        batch_mm_data = {}
        if server_flag:
            if self.sampling_transfer_dock and is_generate:
                td = self.sampling_transfer_dock
            else:
                td = self.td

            if enable_partial_rollout:
                # 获取单条数据，不满足的位置补重复样本
                dp_world_size = self.parallel_state.get_data_parallel_world_size()
                batch_data, index = ray.get(td.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    dp_world_size,
                    indexes=indexes,
                    get_n_samples=get_n_samples,
                    allow_partial_ready_data=True,
                ))
            else:
                batch_data, index = ray.get(
                    td.get_experience.remote(experience_consumer_stage, experience_columns,
                                             experience_count, indexes=indexes,
                                             get_n_samples=get_n_samples,
                                             use_batch_seqlen_balance=self.rl_config.use_dp_batch_balance,
                                             allow_partial_ready_data=False))  # cpu数据
            batch_data = remove_padding_tensor_dict_to_dict(batch_data)
            if not index:  # 判断是否取出数据，未取出数据为-1
                index = [-1] * experience_count
            elif is_multimodal():
                if self.sampling_transfer_dock and is_generate:
                    batch_mm_data = ray.get(
                        self.mm_sampling_transfer_dock.get_experience.remote(
                            experience_consumer_stage,
                            mm_columns,
                            index,
                            get_n_samples
                        )
                    )
                else:
                    batch_mm_data = ray.get(
                        self.mm_td.get_experience.remote(
                            experience_consumer_stage,
                            mm_columns,
                            index,
                            get_n_samples
                        )
                    )

            if not index:
                index = [-1] * experience_count
            send_index_dict = {}
            send_index_dict["index"] = torch.tensor(index + ([-1] * (experience_count - len(index))))
            index_dict = server.dispatch_transfer_docker_data(send_index_dict)
        else:
            index_dict = client.wait_publisher_message()

        if index_dict["index"].tolist()[0] == -1:
            return None, None

        if server_flag:
            batch_data, batch_data_length = pack_experience_columns(experience_consumer_stage, batch_data, 
                experience_count, enable_partial_rollout=enable_partial_rollout
            )
            send_data_size = {}
            data_size = self.compute_batch_data_size(batch_data)
            send_data_size["size"] = torch.tensor(data_size)
            size_dict = server.dispatch_transfer_docker_data(send_data_size)
        else:
            size_dict = client.wait_publisher_message()

        if size_dict["size"].item() < 10 * 1024 * 1024:  # if less than 10M, use zmq
            if server_flag:
                data_dict = server.dispatch_transfer_docker_data(batch_data)
                data_length_dict = server.dispatch_transfer_docker_data(batch_data_length)
            else:
                data_dict = client.wait_publisher_message()
                data_length_dict = client.wait_publisher_message()

            if len(mm_columns) > 0:
                batch_mm_data = self.get_batch_mm_data(batch_mm_data, mm_columns, server_flag, use_vllm)

            if data_dict:
                pad_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
                if is_multimodal():
                    padded_batch_data = unpack_pad_experience(data_dict, data_length_dict, pad_id, 1)
                    batch_mm_data = unpack_mm_experience(batch_mm_data)
                    padded_batch_data.update(batch_mm_data)
                else:
                    padded_batch_data = unpack_pad_experience(data_dict, data_length_dict, pad_id, tp_size * cp_size)

                return padded_batch_data, index_dict["index"].tolist()
        else:
            if not use_vllm:
                rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                            get_context_parallel_rank(self.parallel_state, use_vllm) == 0 and
                            get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)
            else:
                rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                            get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)

            data_hccl = BroadcastHcclParameter(rank_flg, experience_columns,
                                               batch_data, batch_data_length, use_vllm, index_dict["index"])
            index_without_pad = self.broadcast_data_hccl(data_hccl)
            if len(mm_columns) > 0:
                batch_mm_data = self.get_batch_mm_data(batch_mm_data, mm_columns, rank_flg, use_vllm)

            pad_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
            if batch_data:
                if is_multimodal():
                    padded_batch_data = unpack_pad_experience(batch_data, batch_data_length, pad_id, 1)
                    batch_mm_data = unpack_mm_experience(batch_mm_data)
                    padded_batch_data.update(batch_mm_data)
                else:
                    if cp_algo == "megatron_cp_algo":
                        padded_batch_data = unpack_pad_experience(batch_data, batch_data_length,
                                                                  pad_id, 2 * tp_size * cp_size)
                    else:
                        padded_batch_data = unpack_pad_experience(batch_data, batch_data_length,
                                                                  pad_id, tp_size * cp_size)

                return padded_batch_data, index_without_pad

        return {}, []

    @mstx_timer_decorator
    def dispatch_transfer_dock_data(self, experience_consumer_stage,
                                    experience_columns, experience_count, tp_size=1, cp_size=1, cp_algo=None,
                                    use_vllm=False, indexes=None,
                                    get_n_samples=True, enable_partial_rollout=False, is_generate=False):
        if self.zmq_role != ZMQ_ROLE_NONE:
            zmq_parameter = DispatchZmqParameter(experience_consumer_stage,
                                                 experience_columns, experience_count,
                                                 tp_size, cp_size, cp_algo,
                                                 use_vllm, indexes, get_n_samples, enable_partial_rollout, is_generate)
            return self.dispatch_transfer_dock_data_zmq(zmq_parameter)

        pad_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
        if is_multimodal():
            mm_columns = ray.get(self.mm_td.get_columns.remote(experience_consumer_stage))
        else:
            mm_columns = []

        batch_data = {}
        batch_data_length = {}
        batch_mm_data = {}
        # make sure that all ranks in cp/tp/pp group enter dispatch_transfer_dock_data,
        # in case of rank0 get_experience before other ranks judge td.all_consumed

        rank_flg = False
        if not use_vllm:
            rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_context_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)
        else:
            rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)

        if rank_flg:
            if self.sampling_transfer_dock and is_generate:
                td = self.sampling_transfer_dock
            else:
                td = self.td

            if enable_partial_rollout:
                # 获取单条数据，不满足的位置补重复样本
                dp_world_size = self.parallel_state.get_data_parallel_world_size()
                batch_data, index = ray.get(td.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    dp_world_size,
                    indexes=indexes,
                    get_n_samples=get_n_samples,
                    allow_partial_ready_data=True,
                ))
            else:
                batch_data, index = ray.get(
                    td.get_experience.remote(experience_consumer_stage, experience_columns,
                                             experience_count, indexes=indexes,
                                             get_n_samples=get_n_samples,
                                             use_batch_seqlen_balance=self.rl_config.use_dp_batch_balance,
                                             allow_partial_ready_data=False))  # cpu数据
            batch_data = remove_padding_tensor_dict_to_dict(batch_data)
            if not index:  # 判断是否取出数据，未取出数据为-1
                index = [-1] * experience_count
            elif is_multimodal():
                batch_mm_data = ray.get(
                    self.mm_td.get_experience.remote(
                        experience_consumer_stage,
                        mm_columns,
                        index,
                        get_n_samples
                    )
                )

            index = torch.tensor(index + ([-1] * (experience_count - len(index)))).cuda()
        else:
            index = torch.empty(experience_count, device=torch.cuda.current_device(), dtype=torch.int64)

        torch.distributed.broadcast(
            index, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
            group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
        )

        if not use_vllm:
            torch.distributed.broadcast(
                index, get_context_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_context_parallel_group(self.parallel_state, use_vllm)
            )

        torch.distributed.broadcast(
            index, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
            group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
        )

        if index[0].item() == -1:
            return None, None

        if rank_flg:
            batch_data, batch_data_length = pack_experience_columns(experience_consumer_stage, batch_data,
                experience_count,
                enable_partial_rollout=enable_partial_rollout,
            )

        data_hccl = BroadcastHcclParameter(rank_flg, experience_columns,
                                           batch_data, batch_data_length, use_vllm, index)
        index_without_pad = self.broadcast_data_hccl(data_hccl)
        if len(mm_columns) > 0:
            batch_mm_data = self.get_batch_mm_data(batch_mm_data, mm_columns, rank_flg, use_vllm)

        if batch_data:
            if is_multimodal():
                padded_batch_data = unpack_pad_experience(batch_data, batch_data_length, pad_id, 1)
                batch_mm_data = unpack_mm_experience(batch_mm_data)
                padded_batch_data.update(batch_mm_data)
            else:
                if cp_algo == "megatron_cp_algo":
                    padded_batch_data = unpack_pad_experience(batch_data, batch_data_length, pad_id, 2 * tp_size * cp_size)
                else:
                    padded_batch_data = unpack_pad_experience(batch_data, batch_data_length, pad_id, tp_size * cp_size)

            return padded_batch_data, index_without_pad
        else:
            return {}, []

    @mstx_timer_decorator
    def collect_transfer_dock_data(self, output, index, use_vllm=False, is_generate=False, sync=False):
        if is_pipeline_last_stage(self.parallel_state, use_vllm) and get_tensor_model_parallel_rank(self.parallel_state,
                                                                                                    use_vllm) == 0:
            output = {key: value.cpu() if not isinstance(value, List) else value for key, value in output.items()}
            output = padding_dict_to_tensor_dict(output)
            if self.sampling_transfer_dock and is_generate:
                if sync:
                    ray.get(self.sampling_transfer_dock.put_experience.remote(data_dict=output, indexes=index))
                else:
                    self.sampling_transfer_dock.put_experience.remote(data_dict=output, indexes=index)
                if is_multimodal():
                    if sync:
                        ray.get(self.mm_sampling_transfer_dock.put_experience.remote(batch=output, indexes=index))
                    else:
                        self.mm_sampling_transfer_dock.put_experience.remote(batch=output, indexes=index)
            else:
                if sync:
                    ray.get(self.td.put_experience.remote(data_dict=output, indexes=index))
                else:
                    self.td.put_experience.remote(data_dict=output, indexes=index)
                if is_multimodal():
                    if sync:
                        ray.get(self.mm_td.put_experience.remote(batch=output, indexes=index))
                    else:
                        self.mm_td.put_experience.remote(batch=output, indexes=index)

    @mstx_timer_decorator
    def collect_transfer_dock_mm_data(self, output, index, use_vllm=False):
        if is_pipeline_last_stage(self.parallel_state, use_vllm) and get_tensor_model_parallel_rank(self.parallel_state,
                                                                                                    use_vllm) == 0:
            output = {key: value.cpu() if not isinstance(value, List) else value for key, value in output.items()}
            ray.get(self.mm_td.put_experience.remote(batch=output, indexes=index))


    def get_dp_range_indexes(self, experience_count, use_vllm=False, assign_batch_size=None):
        if use_vllm:
            current_dp_rank, dp_world_size = self.get_vllm_dp_rank()
        else:
            current_dp_rank = self.parallel_state.get_data_parallel_rank()
            dp_world_size = self.parallel_state.get_data_parallel_world_size()
        if assign_batch_size is None:
            assign_batch_size = self.megatron_config.global_batch_size // dp_world_size
        return get_current_dp_range_indexes(experience_count=experience_count,
                                            assign_batch_size=assign_batch_size,
                                            current_dp_rank=current_dp_rank)

    @staticmethod
    def get_vllm_dp_rank():
        get_rollout_data_parallel_rank = torch.distributed.get_rank()
        vllm_dp_groups = get_vllm_tp_group_ranks()
        if vllm_dp_groups is None:
            raise ValueError("vllm dp groups is None")
        for index, dp_group in enumerate(vllm_dp_groups):
            if get_rollout_data_parallel_rank in dp_group:
                current_dp_rank = index
        return current_dp_rank, len(vllm_dp_groups)


    def get_batch_mm_data(self, batch_mm_data, mm_columns, rank_flg, use_vllm):
        for key in mm_columns:
            if rank_flg:
                if key not in batch_mm_data.keys():
                    raise KeyError(f'{key} is missing!')
                batch_data_shape = torch.tensor(
                    batch_mm_data[key].shape, dtype=torch.int64, device=torch.cuda.current_device())

                if batch_mm_data[key].dtype == torch.int64:
                    batch_data_dtype = torch.tensor(
                        1, dtype=torch.int64, device=torch.cuda.current_device())
                elif batch_mm_data[key].dtype == torch.bfloat16:
                    batch_data_dtype = torch.tensor(
                        2, dtype=torch.int64, device=torch.cuda.current_device())
                else:
                    batch_data_dtype = torch.tensor(
                        3, dtype=torch.int64, device=torch.cuda.current_device())
            else:
                batch_data_shape = torch.empty(2, device=torch.cuda.current_device(), dtype=torch.int64)
                batch_data_dtype = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)

            # TP domain sync
            torch.distributed.broadcast(
                batch_data_shape, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_dtype, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            # CP domain sync
            if not use_vllm:
                torch.distributed.broadcast(
                    batch_data_shape, get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )
                torch.distributed.broadcast(
                    batch_data_dtype, get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )
            # PP domain sync
            torch.distributed.broadcast(
                batch_data_shape, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_dtype, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )

            if not rank_flg:
                if batch_data_dtype == 1:
                    batch_mm_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                device=torch.cuda.current_device(),
                                                dtype=torch.int64)
                elif batch_data_dtype == 2:
                    batch_mm_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                device=torch.cuda.current_device(),
                                                dtype=torch.bfloat16)
                else:
                    batch_mm_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                device=torch.cuda.current_device(),
                                                dtype=torch.float32)

            # 传输tensor数据
            torch.distributed.broadcast(
                batch_mm_data[key].cuda(), get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_mm_data[key].cuda(), get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )
            if not use_vllm:
                torch.distributed.broadcast(
                    batch_mm_data[key].cuda(), get_context_parallel_src_rank(self.parallel_state, use_vllm),
                    group=get_context_parallel_group(self.parallel_state, use_vllm)
                )
        return batch_mm_data