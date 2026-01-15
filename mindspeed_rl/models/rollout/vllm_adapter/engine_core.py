# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.

import os
import time

from copy import deepcopy
from vllm.config import ParallelConfig, VllmConfig
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHash,
                                         generate_scheduler_kv_cache_config,
                                         get_kv_cache_configs,
                                         get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        UniformTypeKVCacheSpecs)

logger = init_logger(__name__)


def _initialize_kv_caches(
        self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
    start = time.time()

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # Profiles the peak memory usage of the model to determine how much
    # memory can be allocated for kv cache.
    available_gpu_memory = self.model_executor.determine_available_memory()

    if len(kv_cache_specs) != len(available_gpu_memory):
        raise ValueError("not match the number")
    # Get the kv cache tensor size
    self.kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs,
                                            available_gpu_memory)

    # Since we use a shared centralized controller, we need the
    # `kv_cache_config` to be consistent across all workers to make sure
    # all the memory operators can be applied to all workers.
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
        self.kv_cache_configs)

    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    if not all(cfg.num_blocks == self.kv_cache_configs[0].num_blocks for cfg in self.kv_cache_configs):
        raise ValueError("not have the same blocks")
    num_gpu_blocks = self.kv_cache_configs[0].num_blocks
    num_cpu_blocks = 0
    scheduler_kv_cache_config = self.kv_cache_configs[0]

    # Initialize kv cache and warmup the execution

    elapsed = time.time() - start
    logger.info(("init engine (profile, create kv cache, "
                 "warmup model) took %.2f seconds"), elapsed)
    return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config


EngineCore._initialize_kv_caches = _initialize_kv_caches


#此处patch防止在迭代时多次初始化attn_backends
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    """
    Initialize KV cache based on `kv_cache_config`.
    Args:
        kv_cache_config: Configuration for the KV cache, including the KV
        cache size of each layer
    """
    kv_cache_config = deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config
    self.may_add_encoder_only_layers_to_kv_cache_config()
    # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
    if not self.attn_groups:
        self.initialize_attn_backend(kv_cache_config)
    self.use_hybrid_blocks = (len(self.attn_groups) > 1)
    # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
    self.need_accepted_tokens = any([
        isinstance(attn_group[0].kv_cache_spec, MambaSpec)
        for attn_group in self.attn_groups
    ])

    self.may_reinitialize_input_batch(kv_cache_config)

    if self.use_sparse:
        kv_caches = self.initialize_kv_cache_tensors_deepseek_sfa(
            kv_cache_config)
    elif self.model_config.is_deepseek_mla:
        kv_caches = self.initialize_kv_cache_tensors_deepseek_mla(
            kv_cache_config)
    else:
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
    if has_kv_transfer_group():
        get_kv_transfer_group().register_kv_caches(kv_caches)


NPUModelRunner.initialize_kv_cache = initialize_kv_cache