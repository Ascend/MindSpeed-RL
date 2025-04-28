# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import os
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig


def validate_rl_args(
        actor_config: MegatronConfig,
        ref_config: MegatronConfig,
        reward_config: MegatronConfig,
        rl_config: RLConfig,
        generate_config: GenerateConfig
    ):

    # 检查全共卡情况下参数设置
    if rl_config.use_integrated_worker:
        if rl_config.reference_resource is not None:
            raise ValueError(
                f"reference_resource should not be set when use_integrated_worker mode is on.")
        rl_config.reference_resource = rl_config.actor_resource

        if rl_config.reward_resource is not None:
            raise ValueError(
                f" Reward model is not supported when use_integrated_worker mode is on.")

    else:
        if rl_config.integrated_mode_config is not None:
            raise ValueError(
                f"integrated_mode_config should not be set when use_integrated_worker mode is off.")

    # 校验序列长度与模型最大长度
    if generate_config.max_model_len < actor_config.seq_length:
        raise ValueError(
            f"Sequence length exceeds vLLM max_model_len! "
            f"Actor.seq_length={actor_config.seq_length} vs "
            f"GenerateConfig.max_model_len={generate_config.max_model_len}")

    # 校验资源分配合理性
    def _validate_resource(resource, t_size, p_size, c_size, component):
        product = t_size * p_size * c_size
        if resource.num_npus % product != 0:
            raise ValueError(
                f"Invalid {component} resource allocation! "
                f"Resource={resource} must be divisible by (tensor_parallel * pipeline_parallel * context_parallel) = {t_size}*{p_size}*{c_size}={product}")

    _validate_resource(rl_config.actor_resource,
                       actor_config.tensor_model_parallel_size,
                       actor_config.pipeline_model_parallel_size,
                       actor_config.context_parallel_size,
                       "Actor")

    _validate_resource(rl_config.reference_resource,
                       ref_config.tensor_model_parallel_size,
                       ref_config.pipeline_model_parallel_size,
                       ref_config.context_parallel_size,
                       "Reference")
    if rl_config.reward_resource:
        _validate_resource(rl_config.reward_resource,
                           reward_config.tensor_model_parallel_size,
                           reward_config.pipeline_model_parallel_size,
                           reward_config.context_parallel_size,
                           "Reward")
                           
    # 校验批次大小与微批次关系
    def _validate_batch_ratio(global_batch, micro_batch, n_samples, component):
        if (global_batch * n_samples) % micro_batch != 0:
            raise ValueError(
                f"Invalid {component} batch configuration! "
                f"(global_batch_size * n_samples) = {global_batch}*{n_samples} = {global_batch * n_samples} "
                f"must be divisible by micro_batch_size {micro_batch}")

    _validate_batch_ratio(actor_config.global_batch_size,
                          actor_config.micro_batch_size,
                          rl_config.n_samples_per_prompt,
                          "Actor")

    _validate_batch_ratio(ref_config.global_batch_size,
                          ref_config.micro_batch_size,
                          rl_config.n_samples_per_prompt,
                          "Reference")

    if rl_config.reward_resource:
        _validate_batch_ratio(reward_config.global_batch_size,
                              reward_config.micro_batch_size,
                              rl_config.n_samples_per_prompt,
                              "Reward")

    # 校验数据并行与批次关系
    def _validate_data_parallel(global_batch_size, data_parallel, micro_batch_size, n_samples, component):
        if global_batch_size % data_parallel != 0:
            raise ValueError(
                f"{component} global_batch_size {global_batch_size} "
                f"must be divisible by data_parallel_size {data_parallel}")
        
        if (global_batch_size // data_parallel * n_samples) % micro_batch_size != 0:
            raise ValueError(
                f"{component} global_batch_size {actor_config.global_batch_size} "
                f" // data_parallel {data_parallel}  * n_samples {n_samples} "
                f"must be divisible by micro_batch_size {micro_batch_size} ")

    # 计算数据并行度
    actor_data_parallel_size = rl_config.actor_resource.num_npus // (
            actor_config.tensor_model_parallel_size *
            actor_config.pipeline_model_parallel_size *
            actor_config.context_parallel_size)

    generate_config.data_parallel_size = rl_config.actor_resource.num_npus // (
            generate_config.infer_tensor_parallel_size *
            generate_config.infer_pipeline_parallel_size)

    ref_data_parallel_size = rl_config.reference_resource.num_npus // (
            ref_config.tensor_model_parallel_size *
            ref_config.pipeline_model_parallel_size *
            ref_config.context_parallel_size)

    _validate_data_parallel(actor_config.global_batch_size,
                            actor_data_parallel_size,
                            actor_config.micro_batch_size,
                            rl_config.n_samples_per_prompt,
                            "Actor")

    _validate_data_parallel(actor_config.global_batch_size,
                            generate_config.data_parallel_size,
                            generate_config.micro_batch_size,
                            rl_config.n_samples_per_prompt,
                            "Generation")

    if not rl_config.use_integrated_worker:
        _validate_data_parallel(ref_config.global_batch_size,
                                ref_data_parallel_size,
                                ref_config.micro_batch_size,
                                rl_config.n_samples_per_prompt,
                                "Reference")

    if rl_config.reward_resource:
        reward_data_parallel_size = rl_config.reward_resource.num_npus // (
                reward_config.tensor_model_parallel_size *
                reward_config.pipeline_model_parallel_size *
                reward_config.context_parallel_size)

        if not rl_config.use_integrated_worker:
            _validate_data_parallel(reward_config.global_batch_size,
                                    reward_data_parallel_size,
                                    reward_config.micro_batch_size,
                                    rl_config.n_samples_per_prompt,
                                    "Reference")

    # 检查验证器参数匹配
    if len(rl_config.verifier_function) != len(rl_config.verifier_weight):
        raise ValueError(
            f"Verifier function and weight length mismatch: "
            f"{len(rl_config.verifier_function)} vs {len(rl_config.verifier_weight)}")


def validate_data_handler_config(config):
    support_prompt_type_handler = [
        "AlpacaStyleInstructionHandler",
        "AlpacaStylePairwiseHandler",
        "AlpacaStyleProcessRewardHandler",
        "R1AlpacaStyleInstructionHandler",
    ]
    if config.prompt_type is not None and config.handler_name not in support_prompt_type_handler:
        raise ValueError(f'If specify prompt_type , handler name must be in:\n{support_prompt_type_handler}.')

    if (config.merge_group_keys is not None) and (not os.path.isdir(config.input)):
        raise ValueError(f"{config.input} is not a directory or does not exist")

    if not os.path.isdir(os.path.dirname(config.output_prefix)):
        raise ValueError(f"{os.path.dirname(config.output_prefix)} is not a directory or does not exist")

    if not config.pack and config.neat_pack:
        raise ValueError("Require set `pack` True when `neat-pack` is True.")
