# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import os
import re
import subprocess
from abc import ABC
from typing import List, Callable
import socket

import torch
import torch_npu
import ray
import torch.distributed as dist

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
from mindspeed_rl.utils.utils import get_current_dp_range_indexes
from mindspeed_rl.trainer.utils.transfer_dock import pack_experience_columns, unpack_pad_experience
from mindspeed_rl.trainer.utils.mm_transfer_dock import unpack_mm_experience
from mindspeed_rl.utils.utils import mstx_timer_decorator, is_multimodal

logger = Loggers("base_worker")

_DP_RANGE_DATA_CONSUMED_FLAG = 0

_DP_RANGE_DATA_NOT_CONSUMED_FLAG = 1


class BaseRayWorker:
    def __init__(self):
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        self._local_rank = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
        torch.npu.set_device(self._local_rank)
        current_device = torch.npu.current_device()
        if os.environ.get("MASTER_ADDR", 0) == "localhost":
            self._master_addr = self._get_current_node_ip()
            self._master_port = self._get_free_port()
            os.environ["MASTER_ADDR"] = self._master_addr
            os.environ["MASTER_PORT"] = str(self._master_port)
        else:
            self._master_addr = os.environ.get("MASTER_ADDR")
            self._master_port = os.environ.get("MASTER_PORT")
        os.environ["LOCAL_RANK"] = str(self._local_rank)
        logger.info(f"worker init begin, current device id: {current_device}, rank: {self._rank},"
                    f" world_size: {self._world_size}, local rank: {self._local_rank},"
                    f" master_addr: {self._master_addr}, master_port: {self._master_port}")

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    def _get_current_node_ip(self) -> str:
        try:
            # 创建一个 UDP 套接字（仅用于获取接口信息）
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # 连接到一个外部地址（无需真实通信）
                s.connect(("8.8.8.8", 80))  # Google DNS 服务器
                local_ip = s.getsockname()[0]
        except Exception:
            local_ip = self._get_ip_by_ifname()
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

    @staticmethod
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

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
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
        self.args = None

    @mstx_timer_decorator
    def all_consumed(self, experience_consumer_stage, sorted_indexes, use_vllm=False):
        if self.rl_config.guarantee_order and not sorted_indexes:
            return _DP_RANGE_DATA_CONSUMED_FLAG
        elif self.rl_config.guarantee_order:
            return _DP_RANGE_DATA_NOT_CONSUMED_FLAG
        if use_vllm:
            current_device = next(self.inference_model.model.parameters()).device
        else:
            current_device = next(self.model[0].parameters()).device
        status = torch.tensor(0, device=current_device)

        rank_flg = False
        if not use_vllm:
            rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_context_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)
        else:
            rank_flg = (get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and
                        get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0)
        if rank_flg:
            if self.sampling_transfer_dock and ray.get(self.sampling_transfer_dock.get_cur_index.remote()):
                status = torch.tensor(int(not ray.get(self.sampling_transfer_dock.all_consumed.remote(experience_consumer_stage))),
                                      device=current_device)
            else:
                status = torch.tensor(int(not ray.get(self.td.all_consumed.remote(experience_consumer_stage))),
                                      device=current_device)
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

    @mstx_timer_decorator
    def dispatch_transfer_dock_data(self, experience_consumer_stage,
                                    experience_columns, experience_count, tp_size=1, cp_size=1, cp_algo=None,
                                    use_vllm=False, indexes=None,
                                    get_n_samples=True):
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
            if self.sampling_transfer_dock and ray.get(self.sampling_transfer_dock.get_cur_index.remote()):
                batch_data, index = ray.get(
                    self.sampling_transfer_dock.get_experience.remote(experience_consumer_stage, experience_columns,
                                                       experience_count, indexes=indexes,
                                                       get_n_samples=get_n_samples,
                                                       use_batch_seqlen_balance=self.rl_config.use_dp_batch_balance))  # cpu数据
            else:
                batch_data, index = ray.get(
                    self.td.get_experience.remote(experience_consumer_stage, experience_columns,
                                                       experience_count, indexes=indexes,
                                                       get_n_samples=get_n_samples,
                                                       use_batch_seqlen_balance=self.rl_config.use_dp_batch_balance))  # cpu数据
            batch_data = remove_padding_tensor_dict_to_dict(batch_data)
            if not index:  # 判断是否取出数据，未取出数据为-1
                index = [-1] * experience_count
            elif is_multimodal():
                batch_mm_data = ray.get(self.mm_td.get_experience.remote(mm_columns, index, get_n_samples))

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
            self.sampling_transfer_dock.print_consumer_status(experience_consumer_stage, "sampling_transfer_dock")
            self.td.print_consumer_status(experience_consumer_stage, "td")
            return None, None

        if rank_flg:
            batch_data, batch_data_length = pack_experience_columns(batch_data, experience_count)

        for key in experience_columns:
            if rank_flg:
                batch_data_shape = torch.tensor(batch_data[key].shape,
                                                dtype=torch.int64, device=torch.cuda.current_device())

                batch_data_length_shape = torch.tensor(batch_data_length[key].shape, dtype=torch.int64, device=torch.cuda.current_device())

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
            torch.distributed.broadcast(batch_data_length_shape, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
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
                torch.distributed.broadcast(batch_data_length_shape, get_context_parallel_src_rank(self.parallel_state, use_vllm),
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
            torch.distributed.broadcast(batch_data_length_shape, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
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
                batch_data_length[key] = torch.empty(batch_data_length_shape[0], device=torch.cuda.current_device(), dtype=torch.int32)

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

            torch.distributed.broadcast(batch_data_length[key].cuda(), get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                                        group=get_tensor_model_parallel_group(self.parallel_state, use_vllm))

            if not use_vllm:
                torch.distributed.broadcast(batch_data_length[key].cuda(), get_context_parallel_src_rank(self.parallel_state, use_vllm),
                                            group=get_context_parallel_group(self.parallel_state, use_vllm))

            torch.distributed.broadcast(batch_data_length[key].cuda(), get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                                        group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm))

            index_without_pad = index.cpu().numpy().tolist()[:batch_data_shape[0]]

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
    def collect_transfer_dock_data(self, output, index, use_vllm=False):
        if is_pipeline_last_stage(self.parallel_state, use_vllm) and get_tensor_model_parallel_rank(self.parallel_state,
                                                                                                    use_vllm) == 0:
            output = {key: value.cpu() if not isinstance(value, List) else value for key, value in output.items()}
            output = padding_dict_to_tensor_dict(output)
            if self.sampling_transfer_dock and ray.get(self.sampling_transfer_dock.get_cur_index.remote()):
                self.sampling_transfer_dock.put_experience.remote(data_dict=output, indexes=index)
            else:
                self.td.put_experience.remote(data_dict=output, indexes=index)

    @mstx_timer_decorator
    def collect_transfer_dock_mm_data(self, output, index, use_vllm=False):
        if is_pipeline_last_stage(self.parallel_state, use_vllm) and get_tensor_model_parallel_rank(self.parallel_state,
                                                                                                    use_vllm) == 0:
            output = {key: value.cpu() if not isinstance(value, List) else value for key, value in output.items()}
            self.mm_td.put_experience.remote(batch=output, indexes=index)


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
        return batch_mm_data