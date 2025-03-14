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
from typing import Type, Dict, Callable

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
from mindspeed_rl.workers.reference_woker import ReferenceWorker
from mindspeed_rl.workers.reward_woker import RewardWorker


def get_rl_resource_by_worker_type(rl_config: RLConfig, worker: Type[BaseWorker]):
    if (worker.__ray_actor_class__.__name__ ==
            ActorHybridWorker.__ray_actor_class__.__name__):
        return rl_config.actor_resource
    elif (worker.__ray_actor_class__.__name__ ==
          RewardWorker.__ray_actor_class__.__name__):
        return rl_config.reward_resource
    elif (worker.__ray_actor_class__.__name__ ==
          ReferenceWorker.__ray_actor_class__.__name__):
        return rl_config.reference_resource
    else:
        return None


def get_npu_deployment(rl_config: RLConfig, worker: Type[BaseWorker]):
    resource = get_rl_resource_by_worker_type(rl_config, worker)
    if resource is None:
        return 1
    return resource.num_npus


class RayActorGroup:
    def __init__(
            self,
            worker: Type[BaseWorker],
            placement_group: PlacementGroup,
            megatron_config: MegatronConfig,
            rl_config: RLConfig,
            model_provider: Callable,
            initialize_func: Callable,
            parallel_state: ModuleType,
            get_model: Callable = None,
            get_megatron_optimizer: Callable = None,
            get_optimizer_param_scheduler: Callable = None,
            load_checkpoint: Callable = None,
            save_checkpoint: Callable = None,
            get_args: Callable = None,
            tokenizer: BaseTokenizer = None,
            get_forward_backward_func: Callable = None,
            generate_config: GenerateConfig = None,
            resources: Dict[str, float] = None,
            num_resources_per_node: int = None,
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
        parallel_state      : parallel state of actor
        get_model           : model getter
        get_megatron_optimizer          : model megatron optimizer
        get_optimizer_param_scheduler   : model optimizer
        load_checkpoint     : model checkpoint load function
        save_checkpoint     : model checkpoint save function
        get_args            : model args getter
        tokenizer           : tokenizer
        get_forward_backward_func       : model forward backward function
        generate_config     : vllm config data
        resources           : user defined ray resource
        num_resources_per_node  : number of resources per node
        kwargs              : keyword arguments
        """
        self.worker = worker
        self.megatron_config = megatron_config
        self.rl_config = rl_config
        self.generate_config = generate_config
        self.model_provider = model_provider
        self.initialize_func = initialize_func
        self.parallel_state = parallel_state
        self.get_model = get_model
        self.get_megatron_optimizer = get_megatron_optimizer
        self.get_optimizer_param_scheduler = get_optimizer_param_scheduler
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.get_args = get_args
        self.tokenizer = tokenizer
        self.get_forward_backward_func = get_forward_backward_func
        self.kwargs = kwargs
        self.num_npus = get_npu_deployment(rl_config, worker)
        self.resources = resources
        self.num_resources_per_node = num_resources_per_node
        self.actor_handlers = []
        self.temp_actor_ref_objs = []
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

        bundles = [{"NPU": 1, "CPU": 1} for _ in range(self.num_npus)]
        placement_group = ray.util.placement_group(bundles, strategy="PACK")
        ray.get(placement_group.ready())
        return placement_group

    def create_actor_handlers(self, placement_group, world_size, rank_index, master_addr, master_port) \
            -> ray.actor.ActorHandle:
        runtime_env = {
            "env_vars": {
                "MASTER_ADDR": master_addr if master_addr else "localhost",
                "MASTER_PORT": str(master_port) if master_port else "",
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank_index),
            }
        }
        return self.worker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=rank_index
            ),
            runtime_env=runtime_env
        ).remote(
            self.megatron_config,
            self.rl_config,
            self.generate_config,
            model_provider=self.model_provider,
            initialize_func=self.initialize_func,
            parallel_state=self.parallel_state,
            get_model=self.get_model,
            get_megatron_optimizer=self.get_megatron_optimizer,
            get_optimizer_param_scheduler=self.get_optimizer_param_scheduler,
            load_checkpoint=self.load_checkpoint,
            save_checkpoint=self.save_checkpoint,
            get_args=self.get_args,
            tokenizer=self.tokenizer,
            get_forward_backward_func=self.get_forward_backward_func,
            **self.kwargs
        )

    def build_master_actor(self, placement_group, world_size) -> ray.actor.ActorHandle:
        actor_handle = self.create_actor_handlers(
            placement_group, world_size, 0, None, None)
        self.actor_handlers.append(actor_handle)
        return actor_handle

    def build_worker_actor(self, master_handler, placement_group, world_size) -> None:
        master_addr, master_port = ray.get(master_handler.get_master_addr_port.remote())
        for rank in range(1, world_size):
            self.actor_handlers.append(self.create_actor_handlers(
                placement_group, world_size, rank, master_addr, master_port))

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

    def sync_init_transfer_dock(self, transfer_dock):
        for actor in self.actor_handlers:
            ray.get(actor.init_transfer_dock.remote(transfer_dock))

    def wait_all_ref_objs_run_over(self):
        for ref_obj in self.temp_actor_ref_objs:
            ray.get(ref_obj)
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

    def compute_rm_score(self, blocking=False):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.compute_rm_score.remote())
        if blocking:
            ray.get(self.temp_actor_ref_objs)

    def update(self, kl_ctrl):
        actor_train_objs = []
        for actor in self.actor_handlers:
            actor_train_objs.append(actor.update.remote(kl_ctrl))
        return ray.get(actor_train_objs)

    def save_checkpoint(self, iteration):
        actor_train_objs = []
        for actor in self.actor_handlers:
            actor_train_objs.append(actor.save_checkpoint.remote(iteration))
        return ray.get(actor_train_objs)

    def initialize(self):
        for actor in self.actor_handlers:
            self.temp_actor_ref_objs.append(actor.initialize.remote())
        return self

    def get_consumed_train_samples(self):
        return ray.get(self.actor_handlers[0].get_consumed_train_samples.remote())
