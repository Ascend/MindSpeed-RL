# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from functools import partial
import copy
import itertools
import random
import numpy


import pytest
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mindspeed import megatron_adaptor
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.timers import DummyTimer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.training import get_model
import megatron.core.parallel_state as ps

from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from tests.test_tools.dist_test import DistributedTest


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def initialize_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(num_layers=8, hidden_size=512, num_attention_heads=32, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    model = GPTModel(config=transformer_config, transformer_layer_spec=get_gpt_layer_local_spec(), vocab_size=1024, max_sequence_length=64, pre_process=pre_process, post_process=post_process)

    model.bfloat16()
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def init_mock_args(args, use_distributed_optimizer=False, reuse_fp32_param=False):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.bf16 = True
    args.accumulate_allreduce_grads_in_fp32 = True
    args.use_distributed_optimizer = use_distributed_optimizer
    args.ddp_bucket_size = None
    args.reuse_fp32_param = reuse_fp32_param


def setup_model_and_optimizer(seed, use_distributed_optimizer=False):
    model = get_model(partial(initialize_gpt_model, seed=seed))
    set_random_seed(seed)
    config = OptimizerConfig(lr=1e-4, bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=use_distributed_optimizer)
    config.timers = Timers()
    optimizer = get_megatron_optimizer(config, model)

    for group in optimizer.optimizer.param_groups:
        for p in group['params']:
            if len(optimizer.optimizer.state[p]) == 0:
                optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)
    optimizer.reload_model_params()
    return model, optimizer


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


class TestReuseFP32Param(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("is_deterministic", [True, False])
    @pytest.mark.parametrize("tp_pp", [(2, 2), (8, 1)])
    def test_reuse_float16_params_optimizer(self, tp_pp, is_deterministic):
        args = parse_args(None, True)
        args.npu_deterministic = is_deterministic
        set_args(args)
        # truth
        init_mock_args(args)
        initialize_model_parallel(*tp_pp)
        model, optimizer = setup_model_and_optimizer(seed=2)
        megatron_offloader = MegatronOffLoader(
            model,
            optimizer,
            megatron_config=args,
            distributed_optimizer=DistributedOptimizer,
            float16_optimizer_with_float16_params=Float16OptimizerWithFloat16Params)

        self.before_offloader_data_groups = []
        self.after_onloader_data_groups = []

        for _, float16_group in enumerate(optimizer.float16_groups):
            self.before_offloader_data_list = []
            self.after_onloader_data_list = []

            for _, p in enumerate(float16_group):
                self.before_offloader_data_list.append(p.data)
                self.after_onloader_data_list.append(p.data)

            self.before_offloader_data_groups.append(self.before_offloader_data_list)
            self.after_onloader_data_groups.append(self.after_onloader_data_list)

        for _ in range(10):
            for i_index, float16_group in enumerate(optimizer.float16_groups):
                for j_index, p in enumerate(float16_group):
                    self.before_offloader_data_groups[i_index][j_index] = p.data

            megatron_offloader.offload_optimizer()
            megatron_offloader.offload_param()

            megatron_offloader.onload_param()
            megatron_offloader.onload_optimizer()

            for i_index, float16_group in enumerate(optimizer.float16_groups):
                for j_index, p in enumerate(float16_group):
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
                    self.after_onloader_data_groups[i_index][j_index] = p.data
                    if is_deterministic:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0, atol=0)
                    else:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0.005, atol=0.005)

            optimizer.step()

        truth_params = copy.deepcopy(list(itertools.chain(*optimizer.float16_groups)))

        # reuse
        init_mock_args(args, reuse_fp32_param=True)
        args.reuse_fp32_param = True
        reuse_model, reuse_optimizer = setup_model_and_optimizer(seed=2)
        reuse_megatron_offloader = MegatronOffLoader(
            reuse_model,
            reuse_optimizer,
            megatron_config=args,
            distributed_optimizer=DistributedOptimizer,
            float16_optimizer_with_float16_params=Float16OptimizerWithFloat16Params)

        self.before_offloader_data_groups = []
        self.after_onloader_data_groups = []

        for _, float16_group in enumerate(reuse_optimizer.float16_groups):
            self.before_offloader_data_list = []
            self.after_onloader_data_list = []


            for _, p in enumerate(float16_group):
                self.before_offloader_data_list.append(p.data)
                self.after_onloader_data_list.append(p.data)

            self.before_offloader_data_groups.append(self.before_offloader_data_list)
            self.after_onloader_data_groups.append(self.after_onloader_data_list)

        for _ in range(10):
            for i_index, float16_group in enumerate(reuse_optimizer.float16_groups):
                for j_index, p in enumerate(float16_group):
                    self.before_offloader_data_groups[i_index][j_index] = p.data

            reuse_megatron_offloader.offload_optimizer()
            reuse_megatron_offloader.offload_param()

            reuse_megatron_offloader.onload_param()
            reuse_megatron_offloader.onload_optimizer()

            for i_index, float16_group in enumerate(reuse_optimizer.float16_groups):
                for j_index, p in enumerate(float16_group):
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
                    self.after_onloader_data_groups[i_index][j_index] = p.data
                    if is_deterministic:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0, atol=0)
                    else:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0.005, atol=0.005)

            reuse_optimizer.step()

        reuse_params = copy.deepcopy(list(itertools.chain(*reuse_optimizer.float16_groups)))

        for p, reuse_p in zip(truth_params, reuse_params):
            if is_deterministic:
                assert torch.allclose(p.data, reuse_p.data, rtol=0, atol=0)
            else:
                assert torch.allclose(p.data, reuse_p.data, rtol=0.005, atol=0.005)


    @pytest.mark.parametrize("is_deterministic", [True, False])
    @pytest.mark.parametrize("tp_pp", [(2, 2), (8, 1)])
    def test_reuse_distributed_optimizer(self, tp_pp, is_deterministic):
        args = parse_args(None, True)
        args.npu_deterministic = is_deterministic
        set_args(args)

        # truth
        init_mock_args(args, use_distributed_optimizer=True)
        initialize_model_parallel(*tp_pp)
        model, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        megatron_offloader = MegatronOffLoader(
            model,
            optimizer,
            megatron_config=args,
            distributed_optimizer=DistributedOptimizer,
            float16_optimizer_with_float16_params=Float16OptimizerWithFloat16Params)

        self.before_offloader_data_groups = []
        self.after_onloader_data_groups = []

        for _, float16_group in enumerate(optimizer.model_float16_groups):
            self.before_offloader_data_list = []
            self.after_onloader_data_list = []

            for _, p in enumerate(float16_group):
                self.before_offloader_data_list.append(p.data)
                self.after_onloader_data_list.append(p.data)

            self.before_offloader_data_groups.append(self.before_offloader_data_list)
            self.after_onloader_data_groups.append(self.after_onloader_data_list)


        for _ in range(10):
            for i_index, float16_group in enumerate(optimizer.model_float16_groups):
                for j_index, p in enumerate(float16_group):
                    self.before_offloader_data_groups[i_index][j_index] = p.data

            megatron_offloader.offload_optimizer()
            megatron_offloader.offload_param()

            megatron_offloader.onload_param()
            megatron_offloader.onload_optimizer()

            for i_index, float16_group in enumerate(optimizer.model_float16_groups):
                for j_index, p in enumerate(float16_group):
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
                    self.after_onloader_data_groups[i_index][j_index] = p.data

                    if is_deterministic:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0, atol=0)
                    else:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0.005, atol=0.005)

            optimizer.step()

        truth_params = copy.deepcopy(list(itertools.chain(*optimizer.model_float16_groups)))

        # reuse
        init_mock_args(args, use_distributed_optimizer=True, reuse_fp32_param=True)
        initialize_model_parallel(*tp_pp)
        reuse_model, reuse_optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        reuse_megatron_offloader = MegatronOffLoader(
            reuse_model,
            reuse_optimizer,
            megatron_config=args,
            distributed_optimizer=DistributedOptimizer,
            float16_optimizer_with_float16_params=Float16OptimizerWithFloat16Params)

        self.before_offloader_data_groups = []
        self.after_onloader_data_groups = []

        for _, float16_group in enumerate(reuse_optimizer.model_float16_groups):
            self.before_offloader_data_list = []
            self.after_onloader_data_list = []

            for _, p in enumerate(float16_group):
                self.before_offloader_data_list.append(p.data)
                self.after_onloader_data_list.append(p.data)

            self.before_offloader_data_groups.append(self.before_offloader_data_list)
            self.after_onloader_data_groups.append(self.after_onloader_data_list)

        for _ in range(10):
            for i_index, float16_group in enumerate(reuse_optimizer.model_float16_groups):
                for j_index, p in enumerate(float16_group):
                    self.before_offloader_data_groups[i_index][j_index] = p.data

            reuse_megatron_offloader.offload_optimizer()
            reuse_megatron_offloader.offload_param()

            reuse_megatron_offloader.onload_param()
            reuse_megatron_offloader.onload_optimizer()

            for i_index, float16_group in enumerate(reuse_optimizer.model_float16_groups):
                for j_index, p in enumerate(float16_group):
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
                    self.after_onloader_data_groups[i_index][j_index] = p.data
                    if is_deterministic:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0, atol=0)
                    else:
                        assert torch.allclose(self.after_onloader_data_groups[i_index][j_index], self.before_offloader_data_groups[i_index][j_index], rtol=0.005, atol=0.005)

            reuse_optimizer.step()


        reuse_params = copy.deepcopy(list(itertools.chain(*reuse_optimizer.model_float16_groups)))

        for p, reuse_p in zip(truth_params, reuse_params):
            if is_deterministic:
                assert torch.allclose(p.data, reuse_p.data, rtol=0, atol=0)
            else:
                assert torch.allclose(p.data, reuse_p.data, rtol=0.005, atol=0.005)

