#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd.2023-2025. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reversed.
import sys
from unittest.mock import MagicMock
from datetime import timedelta
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mindspeed_rl.utils.utils import init_torch_compile, replace_torch_compile

# Initialize torch.compile global variables to avoid training-related patches affecting vLLM graph mode enabling.
init_torch_compile(torch.compile)

from mindspeed_llm import megatron_adaptor
import megatron
from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.models.gpt import GPTModel
from megatron.training.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.arguments import validate_args
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import _set_random_seed, _init_autoresume, _initialize_tp_communicators
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from mindspeed_llm.training.arguments import parse_args_decorator
from mindspeed_llm.training.initialize import _compile_dependencies

from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from mindspeed_rl.config_cls import MegatronConfig
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils import parse_args_from_config
from mindspeed_rl.utils.utils import seed_all

logger = Loggers('test_resharding')

sampling_config = {
    "num_completions": 1,
    "logprobs": 1,
    "max_tokens": 128,
    "best_of": 1,
    "top_p": 1.0,
    "top_k": 1,
    "min_p": 0.01,
    "temperature": 0.0,
    "detokenize": False,
    "num_beams": 1,
}


def make_megatron_config(args):
    megatron_config = MegatronConfig(
        training_config={'model': 'qwen25-7b', 'use_fused_rmsnorm': True, 'use_mcore_models': True,
                         'sequence_parallel': True, 'use_flash_attn': True, 
                         'no_masked_softmax_fusion': True, 'attention_softmax_in_fp32': True,
                         'no_gradient_accumulation_fusion': True, 'use_fused_swiglu': True,
                         'use_fused_rotary_pos_emb': True, 'bf16': True, 'use_distributed_optimizer': True,
                         'tokenizer_type': args.tokenizer_type, 'tokenizer_name_or_path': args.tokenizer_path, 'global_batch_size': 2,
                         'seq_length': 512, 'save_interval': 10000,
                         'train_iters': 1000, 'distributed_backend': 'nccl', 'no_shared_storage': True,
                         'variable_seq_lengths': True, 'micro_batch_size': 1, 'tensor_model_parallel_size': args.train_tp,
                         'pipeline_model_parallel_size': args.train_pp, 'lr': '1e-7', 'lr_decay_style': 'constant',
                         'min_lr': 0.0, 'weight_decay': 0.0, 'lr_warmup_fraction': 0.0, 'clip_grad': 10000.0,
                         'adam_beta1': 0.9, 'adam_beta2': 0.999, 'initial_loss_scale': 4096, 'finetune': True,
                         'load': args.model_path, 'save': './ckpt', 'no_load_optim': True, 'no_load_rng': True},
        model_config={
            'qwen25-7b': {'use_mcore_models': True, 'num_layers': 28, 'hidden_size': 3584, 'ffn_hidden_size': 18944,
                          'num_attention_heads': 28, 'rotary_base': 1000000, 'max_position_embeddings': 32768,
                          'make_vocab_size_divisible_by': 1, 'padded_vocab_size': 152064,
                          'untie_embeddings_and_output_weights': True, 'add_qkv_bias': True, 'disable_bias_linear': True,
                          'group_query_attention': True, 'num_query_groups': 4, 'attention_dropout': 0.0,
                          'init_method_std': '0.01', 'hidden_dropout': 0.0, 'adam_beta1': 0.9, 'adam_beta2': 0.95,
                          'position_embedding_type': 'rope', 'normalization': 'RMSNorm', 'use_fused_rmsnorm': True,
                          'swiglu': True, 'use_flash_attn': True, 'no_masked_softmax_fusion': True,
                          'attention_softmax_in_fp32': True, 'no_gradient_accumulation_fusion': True,
                          'use_fused_swiglu': True, 'use_fused_rotary_pos_emb': True, 'bf16': True}}
    )
    return megatron_config

optimizer_config = OptimizerConfig()
optimizer_config.lr = 1e-7
optimizer_config.lr_decay_style = "constant"
optimizer_config.min_lr = 0.0
optimizer_config.weight_decay = 0.0
optimizer_config.lr_warmup_fraction = 0.0
optimizer_config.clip_grad = 10000.0
optimizer_config.adam_beta1 = 0.9
optimizer_config.adam_beta2 = 0.999
optimizer_config.initial_loss_scale = 4096
optimizer_config.bf16 = True
optimizer_config.params_dtype = torch.bfloat16
optimizer_config.use_distributed_optimizer = True

prompt_list = [
    "Give three tips for staying healthy.",
    "What are the three primary colors?",
    "Describe the structure of an atom."
]


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
        args_defaults=None,
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
    if args_defaults is None:
        args_defaults = {}

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
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
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
        if mpu.model_parallel_is_initialized():
            logger.info("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
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
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                logger.info(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )


class TestActor():
    def __init__(self, args):
        megatron_config = make_megatron_config(args)
        initialize_megatron(config=megatron_config)
        actor_module = get_model(gpt_model_provider, None)
        if isinstance(actor_module, nn.ModuleList):
            actor_module = [actor_module[0]]
        self.model = actor_module
        self.optimizer = get_megatron_optimizer(optimizer_config, self.model, None, None, 1.0)
        load_checkpoint(actor_module, self.optimizer, None)
        model_config_mock = MagicMock()
        model_config_mock.num_hidden_layers = 28

        self.inference_engine = VLLMInferEngine(
            tokenizer_name_or_path=args.tokenizer_path,
            train_tensor_parallel_size=args.train_tp,
            train_pipeline_parallel_size=args.train_pp,
            train_expert_parallel_size=args.train_ep,
            train_context_parallel_size=args.train_cp,
            infer_tensor_parallel_size=args.infer_tp,
            infer_pipeline_parallel_size=args.infer_pp,
            infer_expert_parallel_size=args.infer_ep,
            sampling_config=sampling_config,
            max_num_seqs=16,
            max_model_len=4096,
            dtype="bfloat16",
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
            enforce_eager=True,
            megatron_config=megatron_config
        )
        self.megatron_offloader = MegatronOffLoader(
            self.model,
            self.optimizer,
            megatron_config=megatron_config,
            distributed_optimizer=DistributedOptimizer,
            float16_optimizer_with_float16_params=Float16OptimizerWithFloat16Params)
        self.sharding_manager = MegatronShardingManager(
            megatron_model=self.model,
            model_config=model_config_mock,
            infer_tensor_parallel_size=args.infer_tp,
            infer_pipeline_parallel_size=args.infer_pp,
            infer_expert_parallel_size=args.infer_ep,
            num_layer_list=None,
            moe_tp_extend_ep=False,
            parallel_state=mpu,
            inference_engine=self.inference_engine,
            optimizer=self.optimizer,
            optimizer_offload=True,
            grad_offload=True,
            train_param_offload=True,
            enable_validate=False,
            megatron_offloader=self.megatron_offloader
        )
        torch.cuda.empty_cache()
        self.tokenizer_path = args.tokenizer_path

    def generate_sequence(self, prompts):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        input_ids = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")["input_ids"].tolist()

        self.sharding_manager.enter_infer_mode()
        outputs = self.inference_engine.generate_sequences(idx_list=input_ids)[0]
        self.sharding_manager.exit_infer_mode()
        rank = torch.distributed.get_rank()

        for output in outputs:
            text = tokenizer.decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            print(f"Rank:{rank},>>>response>>>:{text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test Actor Configuration")
    parser.add_argument("--model-path", type=str, default="./ckpt")
    parser.add_argument("--tokenizer-type", type=str, default="PretrainedFromHF")
    parser.add_argument("--tokenizer-path", type=str, default="./ckpt")
    parser.add_argument("--train-tp", type=int, default=2)
    parser.add_argument("--train-pp", type=int, default=2)
    parser.add_argument("--train-ep", type=int, default=1)
    parser.add_argument("--train_cp", type=int, default=1)
    parser.add_argument("--infer-tp", type=int, default=4)
    parser.add_argument("--infer-pp", type=int, default=1)
    parser.add_argument("--infer-ep", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with replace_torch_compile():
        test_actor = TestActor(args)
        test_actor.generate_sequence(prompt_list)
