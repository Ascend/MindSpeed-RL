# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
"""
description:
    Launch MindSpeed-RL Worker.
    Expose the RayActorGroup class.

launch remote worker task:
    def create_actor_handlers()
        runtime_env = {
            "env_vars": { "MASTER_ADDR": xxx }
        }
        return self.worker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,                    --> control ray resource deployment
                placement_group_bundle_index=rank_index             --> control ray colocate workers
            ),
            runtime_env=runtime_env                                 --> pass environment variables to remote task
        ).remote(...)                                               --> launch remote task
"""

from types import ModuleType
from typing import Type, Dict, Callable, Tuple, List, Optional
from dataclasses import dataclass
import socket
import acl
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.scheduler.scheduler_ms import create_worker_group_scheduler
from mindspeed_rl.workers.scheduler.launcher import get_npu_deployment, get_device_information, construct_placement_groups


@dataclass
class ActorHandlerParamsMs:
    placement_group: PlacementGroup
    world_size: int
    rank_index: int
    bundle_index: int
    master_addr: str
    master_port: int


class RayActorGroupMs:
    def __init__(
            self,
            worker: Type[BaseWorker],
            placement_group: PlacementGroup,
            megatron_config: MegatronConfig,
            rl_config: RLConfig,
            model_provider: Callable,
            initialize_func: Callable,
            profiler_config: Optional[ProfilerConfig] = None,
            msprobe_config: Optional[MsprobeConfig] = None,
            tokenizer: BaseTokenizer = None,
            generate_config: GenerateConfig = None,
            resources: Dict[str, float] = None,
            num_resources_per_node: int = None,
            get_megatron_module: Callable = None,
            **kwargs
    ):
        """
        description:
        ray actor group, all same work type deploy in one group

        parameters:
        worker              : worker class, such as ReferenceWorker
        placement_group     : ray placement group
        megatron_config     : megatron config data
        rl_config           : reinforcement learning config data
        model_provider      : model provider function
        initialize_func     : model initialization function
        tokenizer           : tokenizer
        generate_config     : vllm config data
        resources           : user defined ray resource
        num_resources_per_node  : number of resources per node
        kwargs              : keyword arguments
        """
        self.worker = worker
        self.megatron_config = megatron_config
        self.rl_config = rl_config
        self.generate_config = generate_config
        self.profiler_config = profiler_config
        self.msprobe_config = msprobe_config
        self.model_provider = model_provider
        self.initialize_func = initialize_func
        self.tokenizer = tokenizer
        self.get_megatron_module = get_megatron_module
        self.kwargs = kwargs
        self.num_npus = get_npu_deployment(rl_config, worker)
        self.resources = resources
        self.num_resources_per_node = num_resources_per_node
        self.actor_handlers = []
        self.temp_actor_ref_objs = []
        self.num_devices_per_node, self.num_nodes = (
            get_device_information(self.num_npus))
        self.initialize_actor_handlers(placement_group)

    def initialize_actor_handlers(self, placement_group):
        world_size = self.num_npus
        placement_group = self.get_placement_group(placement_group=placement_group)

        master_actor = self.build_master_actor(placement_group, world_size)
        if world_size > 1:
            self.build_worker_actor(master_actor, placement_group, world_size)

    def get_placement_group(self, placement_group: PlacementGroup = None) -> PlacementGroup:
        if placement_group is not None:
            return placement_group
        return construct_placement_groups(self.num_npus, self.rl_config.num_cpus_for_placement_group,
                                          self.num_devices_per_node, self.num_nodes)

    def create_actor_handlers(self, param: ActorHandlerParamsMs) \
            -> ray.actor.ActorHandle:
        runtime_env = {
            "env_vars": {
                "MASTER_ADDR": param.master_addr if param.master_addr else "localhost",
                "MASTER_PORT": str(param.master_port) if param.master_port else "",
                "WORLD_SIZE": str(param.world_size),
                "RANK": str(param.rank_index),
                "MS_ROLE": "MS_WORKER",
                "MS_WORKER_NUM": str(param.world_size),
                "MS_NODE_ID": str(param.rank_index), 
                "MS_SCHED_HOST": param.master_addr if param.master_addr else "localhost",
                "MS_SCHED_PORT": str(param.master_port) if param.master_port else "",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "USE_RAY": "true"
            }
        }
        return self.worker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=param.placement_group,
                placement_group_bundle_index=param.bundle_index
            ),
            runtime_env=runtime_env
        ).remote(
            self.megatron_config,
            self.rl_config,
            self.generate_config,
            model_provider=self.model_provider,
            get_megatron_module=self.get_megatron_module,
            initialize_func=self.initialize_func,
            profiler_config=self.profiler_config,
            msprobe_config=self.msprobe_config,
            tokenizer=self.tokenizer,
            **self.kwargs
        )

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind((self._get_current_node_ip(), 0))
            return sock.getsockname()[1]

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    def build_master_actor(self, placement_group, world_size) -> ray.actor.ActorHandle:
        self.ms_sched_host = self._get_current_node_ip()
        self.ms_sched_port = self._get_free_port()
        _scheduler_name = f"my_scheduler_{self.ms_sched_port}"
        scheduler_actor = create_worker_group_scheduler(
                name=_scheduler_name,
                world_size=world_size, 
            )
        self.ms_sched_host = ray.get(scheduler_actor._get_current_node_ip.remote())
        self.ms_sched_port = ray.get(scheduler_actor._get_free_port.remote())
        scheduler_actor.init_process_group.remote()
        scheduler_actor.get_status.remote()

        actor_handle = self.create_actor_handlers(
            ActorHandlerParamsMs(placement_group[0], world_size, 0, 0, self.ms_sched_host, self.ms_sched_port))
        self.actor_handlers.append(actor_handle)
        return actor_handle

    def build_worker_actor(self, master_handler, placement_group, world_size) -> None:
        master_addr, master_port = ray.get(master_handler.get_master_addr_port.remote())
        # set first node device
        for rank in range(1, self.num_devices_per_node):
            self.actor_handlers.append(self.create_actor_handlers(
                ActorHandlerParamsMs(placement_group[0], world_size, rank,
                                   rank, master_addr, master_port)))
        # set other node device
        rank_index = self.num_devices_per_node - 1
        for node_index in range(1, self.num_nodes):
            for bundle_index in range(0, self.num_devices_per_node):
                rank_index += 1
                self.actor_handlers.append(self.create_actor_handlers(
                    ActorHandlerParamsMs(placement_group[node_index], world_size, rank_index,
                                       bundle_index, master_addr, master_port)))

    def execute_async_command(self, method_name: str, *args, **kwargs):
        ray_objs = []
        for handler in self.actor_handlers:
            if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                ray_objs.append(getattr(handler, method_name, None).remote(*args, **kwargs))
        return ray_objs

    def execute_sync_command(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_async_command(method_name, *args, **kwargs))

    def async_init_transfer_dock(self, transfer_dock):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.init_transfer_dock.remote(transfer_dock))
        
    def sync_init_transfer_dock(self, transfer_dock, sampling_transfer_dock=None, mm_sampling_transfer_dock=None):
        for actor in self.actor_handlers:
            ray.get(actor.init_transfer_dock.remote(transfer_dock, sampling_transfer_dock, mm_sampling_transfer_dock))

    def enter_infer_mode(self):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.enter_infer_mode.remote())
        ray.get(self.temp_actor_ref_objs)

    def exit_infer_mode(self):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.exit_infer_mode.remote())
        ray.get(self.temp_actor_ref_objs)

    def wait_all_ref_objs_run_over(self):
        ray.get(self.temp_actor_ref_objs)
        self.temp_actor_ref_objs.clear()

    def get_iteration(self):
        return ray.get(self.actor_handlers[0].get_iteration.remote())

    def generate_sequences(self, blocking=False):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.generate_sequences.remote())
        if blocking:
            ray.get(self.temp_actor_ref_objs)

    def compute_log_prob(self, blocking=False):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.compute_log_prob.remote())
        if blocking:
            ray.get(self.temp_actor_ref_objs)

    def compute_ref_log_prob(self, blocking=False):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.compute_ref_log_prob.remote())
        if blocking:
            ray.get(self.temp_actor_ref_objs)

    def compute_rm_score(self, blocking=False):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.compute_rm_score.remote())
        if blocking:
            ray.get(self.temp_actor_ref_objs)

    def update(self, kl_ctrl, skip_actor_log_prob):
        actor_train_objs = []
        for actor in self.actor_handlers:
            actor_train_objs.append(actor.update.remote(kl_ctrl, skip_actor_log_prob))
        return ray.get(actor_train_objs)

    def save_checkpoint(self, iteration):
        actor_train_objs = []
        for actor in self.actor_handlers:
            actor_train_objs.append(actor.save_ckpt.remote(iteration))
        return ray.get(actor_train_objs)

    def initialize(self):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.initialize.remote())
        return self

    def get_consumed_train_samples(self):
        return ray.get(self.actor_handlers[0].get_consumed_train_samples.remote())
