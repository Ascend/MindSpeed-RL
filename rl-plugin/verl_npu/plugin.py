# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from verl_npu.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRolloutPatch


def apply_npu_plugin():
    # Please make sure to apply the patches 
    # in the correct order so that they can 
    # work properly.
    vLLMRolloutPatch.apply_patch()
