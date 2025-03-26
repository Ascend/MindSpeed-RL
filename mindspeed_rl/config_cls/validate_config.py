# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import os


def validate_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config):
    # 校验序列长度与模型最大长度
    if generate_config.max_model_len < actor_config.seq_length:
        raise ValueError(
            f"Sequence length exceeds vLLM max_model_len! "
            f"Actor.seq_length={actor_config.seq_length} vs "
            f"GenerateConfig.max_model_len={generate_config.max_model_len}")

    # 初始化经验计数配置
    rl_config.experience_count_actor = rl_config.experience_count_actor or rl_config.experience_count
    rl_config.experience_count_ref = rl_config.experience_count_ref or rl_config.experience_count
    rl_config.experience_count_reward = rl_config.experience_count_reward or rl_config.experience_count
    rl_config.experience_count_rule_reward = rl_config.experience_count_rule_reward or rl_config.experience_count

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

    # 计算数据并行度
    actor_data_parallel_size = rl_config.actor_resource.num_npus // (
            actor_config.tensor_model_parallel_size *
            actor_config.pipeline_model_parallel_size *
            actor_config.context_parallel_size)

    ref_data_parallel_size = rl_config.reference_resource.num_npus // (
            ref_config.tensor_model_parallel_size *
            ref_config.pipeline_model_parallel_size *
            ref_config.context_parallel_size)

    # 校验微批次经验分配
    def _validate_micro_batch(experience_count, n_samples_per_prompt, data_parallel, micro_batch, component):
        per_device = experience_count * n_samples_per_prompt // data_parallel
        if per_device % micro_batch != 0:
            raise ValueError(
                f"{component} per-device experience count {per_device} "
                f"must be divisible by micro_batch_size {micro_batch}")

    # 校验奖励模型配置
    if rl_config.reward_resource:
        reward_data_parallel_size = rl_config.reward_resource.num_npus // (
                reward_config.tensor_model_parallel_size *
                reward_config.pipeline_model_parallel_size *
                reward_config.context_parallel_size)

        if rl_config.experience_count_reward % reward_data_parallel_size != 0:
            raise ValueError(
                f"Reward experience count {rl_config.experience_count_reward} "
                f"must be divisible by reward_data_parallel_size {reward_data_parallel_size}")

        _validate_micro_batch(rl_config.experience_count_actor,
                              rl_config.n_samples_per_prompt,
                              reward_data_parallel_size,
                              reward_config.micro_batch_size,
                              "Reward")

    # 校验生成配置数据并行度
    generate_config.data_parallel_size = rl_config.actor_resource.num_npus // (
            generate_config.infer_tensor_parallel_size *
            generate_config.infer_pipeline_parallel_size)

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

    _validate_batch_ratio(reward_config.global_batch_size,
                          reward_config.micro_batch_size,
                          rl_config.n_samples_per_prompt,
                          "Reward")

    # 校验经验计数与全局批次关系
    def _validate_experience_ratio(global_batch, experience_count, component):
        if global_batch % experience_count != 0:
            raise ValueError(
                f"{component} global_batch_size {global_batch} "
                f"must be divisible by experience_count {experience_count}")

    _validate_experience_ratio(actor_config.global_batch_size,
                               rl_config.experience_count_actor,
                               "Actor")

    _validate_experience_ratio(ref_config.global_batch_size,
                               rl_config.experience_count_ref,
                               "Reference")

    _validate_experience_ratio(reward_config.global_batch_size,
                               rl_config.experience_count_reward,
                               "Reward")

    _validate_experience_ratio(reward_config.global_batch_size,
                               rl_config.experience_count_rule_reward,
                               "Rule Reward")

    # 校验数据并行与微批次关系
    if actor_config.global_batch_size % actor_data_parallel_size != 0:
        raise ValueError(
            f"Actor global_batch_size {actor_config.global_batch_size} "
            f"must be divisible by data_parallel_size {actor_data_parallel_size}")

    if (actor_config.global_batch_size // actor_data_parallel_size) % actor_config.micro_batch_size != 0:
        raise ValueError(
            f"Actor per-device batch size {actor_config.global_batch_size // actor_data_parallel_size} "
            f"must be divisible by micro_batch_size {actor_config.micro_batch_size}")

    # 校验经验计数与并行度关系
    if rl_config.experience_count_actor % generate_config.data_parallel_size != 0:
        raise ValueError(
            f"Actor experience_count {rl_config.experience_count_actor} "
            f"must be divisible by generate_data_parallel_size {generate_config.data_parallel_size}")

    if rl_config.experience_count_actor % actor_data_parallel_size != 0:
        raise ValueError(
            f"Actor experience_count {rl_config.experience_count_actor} "
            f"must be divisible by actor_data_parallel_size {actor_data_parallel_size}")

    if rl_config.experience_count_ref % ref_data_parallel_size != 0:
        raise ValueError(
            f"Reference experience_count {rl_config.experience_count_ref} "
            f"must be divisible by ref_data_parallel_size {ref_data_parallel_size}")

    _validate_micro_batch(rl_config.experience_count_actor,
                          rl_config.n_samples_per_prompt,
                          actor_data_parallel_size,
                          actor_config.micro_batch_size,
                          "Actor")

    _validate_micro_batch(rl_config.experience_count_ref,
                          rl_config.n_samples_per_prompt,
                          ref_data_parallel_size,
                          ref_config.micro_batch_size,
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
