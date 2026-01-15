# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.


def get_pipeline_model_parallel_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pp_group().rank_in_group
    else:
        return mpu.get_pipeline_model_parallel_rank()


def get_pipeline_model_parallel_src_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pp_group().first_rank
    else:
        return mpu.get_pipeline_model_parallel_first_rank()


def get_pipeline_model_parallel_group(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pp_group().device_group
    else:
        return mpu.get_pipeline_model_parallel_group()


def is_pipeline_last_stage(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pp_group().is_last_rank
    else:
        return mpu.is_pipeline_last_stage()


def get_tensor_model_parallel_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tensor_model_parallel_rank()
    else:
        return mpu.get_tensor_model_parallel_rank()


def get_context_parallel_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        if not hasattr(vpu, "get_context_parallel_rank"):
            vpu = mpu
        return vpu.get_context_parallel_rank()
    else:
        return mpu.get_context_parallel_rank()



def get_tensor_model_parallel_src_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tp_group().first_rank
    else:
        return mpu.get_tensor_model_parallel_src_rank()


def get_context_parallel_src_rank(mpu, use_vllm=False):
    if use_vllm:
        raise NotImplementedError("not implememted yet.")
    else:
        return mpu.get_context_parallel_global_ranks()[0]


def get_tensor_model_parallel_group(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tp_group().device_group
    else:
        return mpu.get_tensor_model_parallel_group()


def get_context_parallel_group(mpu, use_vllm=False):
    if use_vllm:
        raise NotImplementedError("not implememted yet.")
    else:
        return mpu.get_context_parallel_group()


def get_model_parallel_group(mpu, use_vllm=False):
    if use_vllm:
        import vllm
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tp_group().device_group
    else:
        return mpu.get_model_parallel_group()
        

def get_tensor_and_context_parallel_rank(mpu, use_vllm=False):
    """Return caller's rank in the joint tensor-model-parallel and context-parallel group."""
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tensor_model_parallel_rank()
    else:
        return mpu.get_tensor_and_context_parallel_rank()


def get_tensor_and_context_parallel_group(mpu, use_vllm=False):
    """Get the tensor- and context-parallel group the caller rank belongs to."""
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tp_group().device_group
    else:
        return mpu.get_tensor_and_context_parallel_group()