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

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.tokenizer import BaseTokenizer

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.trainer.utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_src_rank,
    get_pipeline_model_parallel_group,
    is_pipeline_last_stage,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
    get_model_parallel_group
)
from mindspeed_rl.utils.compute import set_parallel_state, set_vocab_parallel

logger = Loggers("base_worker")


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
            **kwargs
    ):
        super().__init__()
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        self.rl_config = rl_config
        self.megatron_config = megatron_config
        self.generate_config = generate_config
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
        self.args = None

    def all_consumed(self, experience_consumer_stage, use_vllm=False):
        if use_vllm:
            current_device = next(self.inference_model.model.parameters()).device
        else:
            current_device = next(self.model[0].parameters()).device
        status = torch.tensor(0, device=current_device)
        if get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and \
                get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0:
            status = torch.tensor(int(not ray.get(self.td.all_consumed.remote(experience_consumer_stage))),
                                  device=current_device)
        torch.distributed.all_reduce(status, group=get_model_parallel_group(self.parallel_state, use_vllm),
                                     op=torch.distributed.ReduceOp.MAX)
        return status

    def setup_distributed_rank(self):
        logger.info(f"getenv RANK         : {os.getenv('RANK')}")
        logger.info(f"getenv WORLD_SIZE   : {os.getenv('WORLD_SIZE')}")
        logger.info(f"getenv LOCAL_RANK   : {os.getenv('LOCAL_RANK')}")
        logger.info(f"getenv MASTER_ADDR  : {os.getenv('MASTER_ADDR')}")
        logger.info(f"getenv MASTER_PORT  : {os.getenv('MASTER_PORT')}")
        logger.info(f"ray alloc NPU ID    :  {int(ray.get_runtime_context().get_accelerator_ids()['NPU'][0])}")
        self.initialize_func(config=self.megatron_config)
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

    def init_transfer_dock(self, td):
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

    def empty_cache(self):
        """Clear GPU cache (can be overridden by subclasses)"""
        torch.cuda.empty_cache()

    def dispatch_transfer_dock_data(self, experience_consumer_stage,
                                    experience_colums, experience_count, n_samples_per_prompt=1, tp_size=1,
                                    use_vllm=False):
        pad_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod

        batch_data = {}
        # make sure that all ranks in tp/pp group enter dispatch_transfer_dock_data,
        # in case of rank0 get_experience before other ranks judge td.all_consumed
        if get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and \
                get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0:
            batch_data, index = ray.get(self.td.get_experience.remote(experience_consumer_stage, experience_colums,
                                                                      experience_count, pad_id=pad_id,
                                                                      multiple=tp_size))  # cpu数据
            if not index:  # 判断是否取出数据，未取出数据为-1
                index = [-1] * experience_count

            index = torch.tensor(index).cuda()
        else:
            index = torch.empty(experience_count, device=torch.cuda.current_device(), dtype=torch.int64)

        #  # 传输index, 并判断是否取出了数据
        torch.distributed.broadcast(
            index, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
            group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
        )
        torch.distributed.broadcast(
            index, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
            group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
        )

        if index[0].item() == -1:
            return None, None

        for key in experience_colums:
            if get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0 and \
                    get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) == 0:
                batch_data_shape = torch.tensor(batch_data[key].shape,
                                                dtype=torch.int64, device=torch.cuda.current_device())

                if batch_data[key].dtype == torch.int64:
                    batch_data_dtype = torch.tensor(1,
                                                    dtype=torch.int64, device=torch.cuda.current_device())
                else:
                    batch_data_dtype = torch.tensor(2,
                                                    dtype=torch.int64, device=torch.cuda.current_device())
            else:
                batch_data_shape = torch.empty(2, device=torch.cuda.current_device(), dtype=torch.int64)
                batch_data_dtype = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)

            # 传输tensor数据形状和类型
            torch.distributed.broadcast(
                batch_data_shape, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_dtype, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_shape, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data_dtype, get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )

            if get_tensor_model_parallel_rank(self.parallel_state, use_vllm) != 0 or \
                    get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) != 0:
                if batch_data_dtype == 1:
                    batch_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                  device=torch.cuda.current_device(),
                                                  dtype=torch.int64)
                else:
                    batch_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                  device=torch.cuda.current_device(),
                                                  dtype=torch.float32)

            # 传输tensor数据
            torch.distributed.broadcast(
                batch_data[key].cuda(), get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
            )
            torch.distributed.broadcast(
                batch_data[key].cuda(), get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
            )
        index = index.cpu().numpy().tolist()
        return batch_data, index

    def collect_transfer_dock_data(self, output, index, n_samples_per_prompt=1, use_vllm=False):
        if is_pipeline_last_stage(self.parallel_state, use_vllm) and get_tensor_model_parallel_rank(self.parallel_state,
                                                                                                    use_vllm) == 0:
            output = {key: value.cpu() if not isinstance(value, List) else value for key, value in output.items()}
            self.td.put_experience.remote(data_dict=output, indexes=index, num_responses=n_samples_per_prompt)
