# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from verl_npu.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRolloutPatch
from verl_npu.utils.seqlen_balancing import SeqlenBalancingPatch
from .transformers_npu.qwen2_patch import apply_qwen2_patch
from .transformers_npu.qwen3_moe_patch import apply_qwen3_moe_patch
from .transformers_npu.npu_flash_attention_patch import apply_npu_flash_attention_patch



def apply_npu_plugin():
    # Please make sure to apply the patches 
    # in the correct order so that they can 
    # work properly.
    vLLMRolloutPatch.apply_patch()
    SeqlenBalancingPatch.apply_patch()
    apply_qwen2_patch()
    apply_qwen3_moe_patch()
    apply_npu_flash_attention_patch()
    
    # In verl, the driver process aggregates the computation results of workers via Ray. 
    # Therefore, after a worker completes its computation job, it will package the output 
    # using tensordict and transfer it to the CPU. Since the `to` operation of tensordict 
    # is non-blocking, when transferring data from a device to the CPU, it is necessary to 
    # ensure that a batch of data has been completely transferred before being used on the 
    # host; otherwise, unexpected precision issues may arise. Tensordict has already noticed 
    # this problem and fixed it.
    # However, the relevant modifications only cover CUDA and MPS devices and do not take effect 
    # for third-party devices such as NPUs. This patch fixes this issue, and the relevant 
    # modifications can be removed once the fix is merged into tensordict.

    from tensordict.base import TensorDictBase

    def _sync_all_patch(self):
        import torch
        from torch._utils import _get_available_device_type, _get_device_module
        try:
            from torch.compiler import is_compiling
        except ImportError:  # torch 2.0
            from torch._dynamo import is_compiling

        device_type = _get_available_device_type()
        if device_type is None:
            return

        if device_type == "cuda":
            if not is_compiling() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        else:
            device_module = _get_device_module(device_type)
            device_module.synchronize()

    TensorDictBase._sync_all = _sync_all_patch