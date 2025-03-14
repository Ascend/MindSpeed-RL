# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from datetime import timedelta
from typing import Dict

import hydra
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from mindspeed_rl.utils import get_tokenizer
from mindspeed_rl.trainer.utils import GRPOTransferDock
from mindspeed_rl.utils import Loggers
from mindspeed_rl.utils.utils import parse_args_from_config, seed_all

logger = Loggers("test_actor_hybrid_worker")


def parse_training_config(config: Dict):
    """
    解析训练配置，提取 actor、rl 和 generate 的配置。

    :param config: 输入的全局配置字典。
    :return: 包含 actor_config、rl_config 和 generate_config 的实例。
    """
    from mindspeed_rl.config_cls.megatron_config import MegatronConfig
    from mindspeed_rl.config_cls.rl_config import RLConfig
    from mindspeed_rl.config_cls.generate_config import GenerateConfig
    actor_config = MegatronConfig({**config.get("megatron_training"), **config.get("actor_config")},
                                  config.get('model'))
    rl_config = RLConfig(config.get("rl_config"))
    generate_config = GenerateConfig(config.get("generate_config"))

    return actor_config, rl_config, generate_config


@ray.remote
def train(config):
    import sys

    origin_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]
    parse_args_from_config(config)
    from mindspeed_llm import megatron_adaptor
    sys.argv = origin_sys_argv

    import megatron
    from megatron.core import parallel_state
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.training import get_optimizer_param_scheduler
    from megatron.training import get_args
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.legacy.model import Float16Module
    from megatron.training.training import get_model, unwrap_model
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training.arguments import core_transformer_config_from_args, validate_args
    from megatron.training.checkpointing import load_args_from_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import _set_random_seed, _init_autoresume, _compile_dependencies, \
        _initialize_tp_communicators

    from mindspeed_llm.training.arguments import parse_args_decorator

    def gpt_model_provider(pre_process, post_process):
        """
        Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
            Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        args = get_args()

        logger.info('building GPT model ...')
        # Experimental loading arguments from configs
        config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )

        return model

    def initialize_megatron(
            extra_args_provider=None,
            args_defaults={},
            ignore_unknown_args=False,
            allow_no_cuda=False,
            skip_mpu_initialization=False,
            config=None,
    ):
        """Set global variables, initialize distributed, and
        set autoresume and random seeds.
        `allow_no_cuda` should not be set unless using megatron for cpu only
        data processing. In general this arg should not be set unless you know
        what you are doing.
        Returns a function to finalize distributed env initialization
        (optionally, only when args.lazy_mpu_init == True)
        """
        if not allow_no_cuda:
            if not torch.cuda.is_available():
                raise ValueError("Megatron requires CUDA.")

        origin_sys_argv = sys.argv
        sys.argv = [sys.argv[0]]
        parse_args_from_config(config)
        parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
        args = parse_args(extra_args_provider, ignore_unknown_args)
        sys.argv = origin_sys_argv

        if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
            if args.load is None:
                raise ValueError("--use-checkpoints-args requires --load argument.")
            load_args_from_checkpoint(args)

        validate_args(args, args_defaults)

        set_global_variables(args)

        if args.use_deter_comp:
            seed_all(args.seed)
            logger.info("deterministic computing is applied for npu.")

        # torch.distributed initialization
        def finish_mpu_init():
            args = get_args()
            # Pytorch distributed.
            _initialize_distributed()

            # Random seeds for reproducibility.
            if args.rank == 0:
                logger.info("> setting random seeds to {} ...".format(args.seed))
            _set_random_seed(args.seed, args.data_parallel_random_init)

        if skip_mpu_initialization:
            return None

        args = get_args()
        if args.lazy_mpu_init:
            args.use_cpu_initialization = True
            # delayed initialization of DDP-related stuff
            # We only set basic DDP globals
            parallel_state.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            # and return function for external DDP manager
            # to call when it has DDP initialized
            parallel_state.set_tensor_model_parallel_rank(args.rank)
            return finish_mpu_init
        else:
            # Megatron's MPU is the master. Complete initialization right away.
            finish_mpu_init()

            # Autoresume.
            _init_autoresume()

            # Compile dependencies.
            _compile_dependencies()

            if args.tp_comm_overlap:
                _initialize_tp_communicators()

            # No continuation function
            return None

    def _initialize_distributed():
        """Initialize torch.distributed and core model parallel."""
        args = get_args()

        device_count = torch.cuda.device_count()
        if torch.distributed.is_initialized():
            if args.rank == 0:
                logger.info("torch distributed is already initialized, skipping initialization...")
            args.rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
        else:
            if args.rank == 0:
                logger.info("> initializing torch distributed...")
            # Manually set the device ids.
            if device_count > 0:
                if args.stage in ["ray_ppo", "ray_online_dpo", "ray_grpo"]:
                    allocated_device = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
                    torch.cuda.set_device(allocated_device)
                else:
                    device = args.rank % device_count
                    if args.local_rank is not None:
                        if args.local_rank != device:
                            raise ValueError("expected local-rank to be the same as rank % device-count.")
                    else:
                        args.local_rank = device
                    torch.cuda.set_device(device)
            # Call the init process
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(minutes=args.distributed_timeout_minutes),
            )

        # Set the tensor model-parallel, pipeline model-parallel, and
        # data-parallel communicators.
        if device_count > 0:
            if parallel_state.model_parallel_is_initialized():
                logger.info("model parallel is already initialized")
            else:
                parallel_state.initialize_model_parallel(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    context_parallel_size=args.context_parallel_size,
                    expert_model_parallel_size=args.expert_model_parallel_size,
                    distributed_timeout_minutes=args.distributed_timeout_minutes,
                    nccl_communicator_config_path=args.nccl_communicator_config_path,
                    order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',
                )
                if args.rank == 0:
                    logger.info(
                        f"> initialized tensor model parallel with size "
                        f"{parallel_state.get_tensor_model_parallel_world_size()}"
                    )
                    logger.info(
                        f"> initialized pipeline model parallel with size "
                        f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                    )

    from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker

    actor_config, rl_config, generate_config = parse_training_config(config)
    bundles = [{"NPU": 1, "CPU": 1}]
    pg = ray.util.placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    runtime_env = {
        "env_vars": {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "",
            "WORLD_SIZE": '1',
            "RANK": '0',
        }
    }
    tokenizer = get_tokenizer(tokenizer_model=actor_config.tokenizer_name_or_path)

    megatron_module = {
        'get_model': get_model,
        'model_provider': gpt_model_provider,
        'initialize_func': initialize_megatron,
        'parallel_state': parallel_state,
        'get_megatron_optimizer': get_megatron_optimizer,
        'get_optimizer_param_scheduler': get_optimizer_param_scheduler,
        'load_checkpoint': load_checkpoint,
        'save_checkpoint': save_checkpoint,
        'get_args': get_args,
        'get_forward_backward_func': get_forward_backward_func,
        'float16_module': Float16Module,
        'unwrap_model': unwrap_model,
        'local_ddp': LocalDDP,
        'distributed_data_parallel_config': DistributedDataParallelConfig,
    }

    actor_worker = ActorHybridWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0
        ),
        runtime_env=runtime_env
    ).remote(
        actor_config,
        rl_config,
        generate_config,
        tokenizer=tokenizer,
        **megatron_module
    )

    ray.get(actor_worker.initialize.remote())

    transfer_dock = GRPOTransferDock.remote(actor_config.global_batch_size)
    ray.get(actor_worker.init_transfer_dock.remote(transfer_dock))

    logger.info("==================== hybrid worker initialize test passed! ====================")


@hydra.main(config_path='../configs', config_name='test_actor_hybrid_worker', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "True",
                         'TOKENIZERS_PARALLELISM': 'true',
                         'NCCL_DEBUG': 'WARN'}})

    ray.get(train.remote(config))


if __name__ == '__main__':
    main()
