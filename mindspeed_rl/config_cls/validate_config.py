# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import os
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.utils.utils import get_node_nums


def validate_rl_args(
        actor_config: MegatronConfig,
        ref_config: MegatronConfig,
        reward_config: MegatronConfig,
        rl_config: RLConfig,
        generate_config: GenerateConfig,
        critic_config: MegatronConfig = None,
        vit_config: MegatronConfig = None
    ):

    #检查后端参数设置
    if hasattr(actor_config, "ai_framework"):
        ai_framework = actor_config.ai_framework
        if ai_framework is not None and ai_framework != "mindspore":
            raise ValueError(f"Invalid value for ai_framework: '{ai_framework}'. Only None or mindspore are allowed")

    #检查训推数据类型是否合规
    if actor_config.bf16 is False or generate_config.dtype != "bfloat16":
        raise ValueError(
                f" megatron_config.bf16 should be true and generate_config.dtype should be bfloat16.")


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

    if actor_config.context_parallel_size > 1 and actor_config.context_parallel_algo is not None:
        if actor_config.context_parallel_algo not in ["ulysses_cp_algo", "megatron_cp_algo"]:
            raise ValueError("Now just support ulysses CP and megatron cp(ring)")
    if actor_config.cp_attention_mask_type not in ["causal"]:
        raise ValueError("Now just support causal cp_attention_mask_type")
    if actor_config.context_parallel_algo == "megatron_cp_algo" and actor_config.context_parallel_size > 1 and rl_config.use_remove_padding:
        if not actor_config.reset_attention_mask:
            raise ValueError("when use ring cp and remove_padding, reset_attention_mask must be true")
    if actor_config.context_parallel_size <= 1 or actor_config.context_parallel_algo != "megatron_cp_algo" or not rl_config.use_remove_padding:
        if actor_config.reset_attention_mask:
            raise ValueError("Just when use ring cp >=2 with remove_padding, reset_attention_mask must be true; otherwise should be false")

    # 校验移除填充特性相关配置
    if rl_config.use_remove_padding:
        if actor_config.pipeline_model_parallel_size > 1 and not actor_config.variable_seq_lengths:
            raise ValueError(
                "'use_remove_padding' feature requires 'variable_seq_lengths=True' when using pipeline parallelism!"
                "If you want to use context parallelism under this premise and encounter the mindspeed_llm validation error about variable_seq_lengths, "
                "you just need to delete the validation code of mindspeed_llm, and it will not cause problems.")

        if not actor_config.reset_position_ids:
            raise ValueError(
                "'use_remove_padding' feature requires 'reset_position_ids=True'! ")
            
        if rl_config.is_multimodal:
            raise ValueError(
                "'multimodal' models cannot use 'use_remove_padding' feature! "
                "Please set 'use_remove_padding=False' in the RLConfig.")

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

    if ref_config:
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

    if ref_config:
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
        actor_config.context_parallel_size
    )

    if rl_config.colocate_actor_and_vit:
        vit_data_parallel_size = rl_config.vit_resource.num_npus // (
            vit_config.tensor_model_parallel_size *
            vit_config.pipeline_model_parallel_size *
            vit_config.context_parallel_size)

    generate_config.data_parallel_size = rl_config.actor_resource.num_npus // (
        generate_config.infer_tensor_parallel_size *
        generate_config.infer_pipeline_parallel_size
    )

    if rl_config.critic_resource:
        critic_data_parallel_size = rl_config.critic_resource.num_npus // (
            critic_config.tensor_model_parallel_size *
            critic_config.pipeline_model_parallel_size *
            critic_config.context_parallel_size)

    if generate_config.infer_pipeline_parallel_size > 1:
        raise ValueError(
            "pipeline_parallel for vllm is not supported yet ! ")

    if ref_config:
        ref_data_parallel_size = rl_config.reference_resource.num_npus // (
            ref_config.tensor_model_parallel_size *
            ref_config.pipeline_model_parallel_size *
            ref_config.context_parallel_size
        )

    _validate_data_parallel(actor_config.global_batch_size,
                            actor_data_parallel_size,
                            actor_config.micro_batch_size,
                            rl_config.n_samples_per_prompt,
                            "Actor")

    rl_config.actor_rollout_dispatch_size = (
        rl_config.actor_rollout_dispatch_size or
        (actor_config.global_batch_size * rl_config.n_samples_per_prompt // generate_config.data_parallel_size)
    )
    _validate_data_parallel(actor_config.global_batch_size,
                            generate_config.data_parallel_size,
                            rl_config.actor_rollout_dispatch_size,
                            rl_config.n_samples_per_prompt,
                            "Generation")

    if not rl_config.use_integrated_worker and ref_config:
        _validate_data_parallel(ref_config.global_batch_size,
                                ref_data_parallel_size,
                                ref_config.micro_batch_size,
                                rl_config.n_samples_per_prompt,
                                "Reference")

    if rl_config.reward_resource:
        reward_data_parallel_size = rl_config.reward_resource.num_npus // (
            reward_config.tensor_model_parallel_size *
            reward_config.pipeline_model_parallel_size *
            reward_config.context_parallel_size
        )

        if not rl_config.use_integrated_worker:
            _validate_data_parallel(reward_config.global_batch_size,
                                    reward_data_parallel_size,
                                    reward_config.micro_batch_size,
                                    rl_config.n_samples_per_prompt,
                                    "Reward")

    # 初始化经验计数配置
    if rl_config.filter_groups_enable:
        # 若开启dapo动态采样，logp和adv的gbs=filter_groups_train_batch_size
        _validate_data_parallel(rl_config.filter_groups_train_batch_size,
                        actor_data_parallel_size,
                        actor_config.micro_batch_size,
                        rl_config.n_samples_per_prompt,
                        "ActorForDapo")
                        
        rl_config.actor_logprob_dispatch_size = (
            rl_config.actor_logprob_dispatch_size or
            (rl_config.filter_groups_train_batch_size * rl_config.n_samples_per_prompt // actor_data_parallel_size)
        )
        rl_config.adv_dispatch_size = (
            rl_config.adv_dispatch_size or
            (rl_config.filter_groups_train_batch_size * rl_config.n_samples_per_prompt)
        )

        num_process = get_node_nums()
        rl_config.dynamic_sampling_dispatch_size = (
            rl_config.dynamic_sampling_dispatch_size or
            (reward_config.global_batch_size * rl_config.n_samples_per_prompt // num_process)
        )
    else:
        rl_config.actor_logprob_dispatch_size = (
            rl_config.actor_logprob_dispatch_size or
            (actor_config.global_batch_size * rl_config.n_samples_per_prompt // actor_data_parallel_size)
        )
        rl_config.adv_dispatch_size = (
            rl_config.adv_dispatch_size or (actor_config.global_batch_size * rl_config.n_samples_per_prompt)
        )

    if ref_config:
        rl_config.ref_dispatch_size = (
            rl_config.ref_dispatch_size or
            (ref_config.global_batch_size * rl_config.n_samples_per_prompt // ref_data_parallel_size)
        )
    
    if rl_config.reuse_image_embeds:
        if rl_config.colocate_actor_and_vit:
            rl_config.actor_image_embeds_dispatch_size = (
                rl_config.actor_image_embeds_dispatch_size or
                (vit_config.global_batch_size * rl_config.n_samples_per_prompt // vit_data_parallel_size)
            )
        else:
            rl_config.actor_image_embeds_dispatch_size = (
                rl_config.actor_image_embeds_dispatch_size or
                (actor_config.global_batch_size * rl_config.n_samples_per_prompt // actor_data_parallel_size)
            )

    if rl_config.reward_resource:
        reward_data_parallel_size = rl_config.reward_resource.num_npus // (
            reward_config.tensor_model_parallel_size *
            reward_config.pipeline_model_parallel_size *
            reward_config.context_parallel_size)
        rl_config.reward_dispatch_size = (
            rl_config.reward_dispatch_size or
            (reward_config.global_batch_size * rl_config.n_samples_per_prompt // reward_data_parallel_size)
        )
    else:
        rule_reward_num_process = get_node_nums()
        rl_config.reward_dispatch_size = (
            rl_config.reward_dispatch_size or (reward_config.global_batch_size * rl_config.n_samples_per_prompt // rule_reward_num_process)
        )
        if reward_config.global_batch_size % rule_reward_num_process != 0:
            raise ValueError(
                f"Reward dispatch size configuration error!"
                f"global_batch_size {reward_config.global_batch_size} must be divisible by the number of nodes in the ray cluster")

    if rl_config.critic_resource:
        rl_config.critic_update_dispatch_size = (
            rl_config.critic_update_dispatch_size or
            (critic_config.global_batch_size * rl_config.n_samples_per_prompt // critic_data_parallel_size)
        )
        rl_config.critic_value_dispatch_size = (
            rl_config.critic_value_dispatch_size or
            (critic_config.global_batch_size * rl_config.n_samples_per_prompt // critic_data_parallel_size)
        )
        rl_config.kl_dispatch_size = (
            rl_config.kl_dispatch_size or (critic_config.global_batch_size * rl_config.n_samples_per_prompt)
        )

    # 校验经验计数与全局批次关系
    def _validate_experience_ratio(global_batch, experience_count, component):
        if global_batch * rl_config.n_samples_per_prompt % experience_count != 0:
            raise ValueError(
                f"{component} global_batch_size {global_batch} "
                f"must be divisible by experience_count {experience_count}")

    if rl_config.filter_groups_enable:
        _validate_experience_ratio(rl_config.filter_groups_train_batch_size,
                                   rl_config.actor_logprob_dispatch_size,
                                   "Actor Infer")
        _validate_experience_ratio(rl_config.filter_groups_train_batch_size,
                                   rl_config.adv_dispatch_size,
                                   "Advantages")

        _validate_experience_ratio(reward_config.global_batch_size,
                                   rl_config.dynamic_sampling_dispatch_size,
                                   "Dynamic Sampling")
        if rl_config.dynamic_sampling_dispatch_size % rl_config.n_samples_per_prompt != 0:
            raise ValueError(
                f"dynamic sampling dispatch size {rl_config.dynamic_sampling_dispatch_size} "
                f"must be divisible by n samples per prompt {rl_config.n_samples_per_prompt}")
    else:
        _validate_experience_ratio(actor_config.global_batch_size,
                                   rl_config.actor_logprob_dispatch_size,
                                   "Actor Infer")
        _validate_experience_ratio(actor_config.global_batch_size,
                                   rl_config.adv_dispatch_size,
                                   "Advantages")

    if ref_config:
        _validate_experience_ratio(ref_config.global_batch_size,
                                rl_config.ref_dispatch_size,
                                "Reference")

    if rl_config.reward_resource:
        _validate_experience_ratio(reward_config.global_batch_size,
                                   rl_config.reward_dispatch_size,
                                   "Reward")
    else:
        _validate_experience_ratio(reward_config.global_batch_size,
                                   rl_config.reward_dispatch_size,
                                   "Rule Reward")

    if rl_config.critic_resource:
        _validate_experience_ratio(critic_config.global_batch_size,
                                   rl_config.critic_value_dispatch_size,
                                   "Critic Infer")
        _validate_experience_ratio(critic_config.global_batch_size,
                                   rl_config.kl_dispatch_size,
                                   "KL")
        _validate_experience_ratio(critic_config.global_batch_size,
                                   rl_config.critic_update_dispatch_size,
                                   "Critic Update")

    # 若指定了自定义的actor_update_dispatch_size，检查 actor_update_dispatch_size 是否符合 on_policy/off_policy 策略要求
    if rl_config.actor_update_dispatch_size:
        if rl_config.actor_update_dispatch_size < rl_config.mini_batch_size / actor_data_parallel_size:
            raise ValueError(
                f"actor_update_dispatch_size={rl_config.actor_update_dispatch_size} "
                f"must be >= mini_batch_size/actor_data_parallel_size "
                f"({rl_config.mini_batch_size}/{actor_data_parallel_size}="
                f"{int(rl_config.mini_batch_size/actor_data_parallel_size)})"
            )

    if rl_config.filter_groups_enable:
        # 若开启dapo动态采样，update的gbs=filter_groups_train_batch_size
        rl_config.actor_update_dispatch_size = (
            rl_config.actor_update_dispatch_size or
            (rl_config.filter_groups_train_batch_size * rl_config.n_samples_per_prompt // actor_data_parallel_size)
        )
        _validate_experience_ratio(rl_config.filter_groups_train_batch_size,
                                   rl_config.actor_update_dispatch_size,
                                   "Actor Update")
    else:
        rl_config.actor_update_dispatch_size = (
            rl_config.actor_update_dispatch_size or
            (actor_config.global_batch_size * rl_config.n_samples_per_prompt // actor_data_parallel_size)
        )
        _validate_experience_ratio(actor_config.global_batch_size,
                                   rl_config.actor_update_dispatch_size,
                                   "Actor Update")

    # 检查验证器参数匹配
    if len(rl_config.verifier_function) != len(rl_config.verifier_weight):
        raise ValueError(
            f"Verifier function and weight length mismatch: "
            f"{len(rl_config.verifier_function)} vs {len(rl_config.verifier_weight)}")

    # DAPO Response Overlong 校验
    if rl_config.overlong_buffer_enable:
        max_tokens = generate_config.sampling_config["max_tokens"]
        if rl_config.overlong_buffer >= max_tokens:
            raise ValueError(
                f"Response max length {max_tokens} "
                f"must greater than Dapo overlong buffer {rl_config.overlong_buffer}")
        if rl_config.rollout_max_tokens != max_tokens:
            raise ValueError(
                f"overlong rollout_max_tokens and generate rollout_max_tokens mismatch: "
                f"{rl_config.rollout_max_tokens} vs {max_tokens}")

    # DAPO Clip Higher 校验
    if rl_config.clip_higher_enable:
        if rl_config.clip_ratio_low > rl_config.clip_ratio_high:
            raise ValueError(
                f"clip_ratio_low {rl_config.clip_ratio_low} "
                f"must less than clip_ratio_high {rl_config.clip_ratio_high}")

    # DAPO filter metric key值校验
    if rl_config.filter_groups_enable:
        metric = rl_config.filter_groups_metric
        verifier_function = rl_config.verifier_function
        if metric not in verifier_function:
            raise ValueError(
                f"filter_groups_metric {metric} must in verifier_function {verifier_function}")
        rl_config.filter_groups_metric += "_rewards/mean"

    # partial_rollout开启时，再开保序会报错
    if rl_config.partial_rollout_max_split > 1 and rl_config.guarantee_order:
        raise ValueError(
            f"guarantee_order must be false when partial_rollout_max_split > 1")


def validate_data_handler_config(config):
    support_prompt_type_handler = [
        "AlpacaStyleInstructionHandler",
        "AlpacaStylePairwiseHandler",
        "AlpacaStyleProcessRewardHandler",
        "R1AlpacaStyleInstructionHandler",
        "Math17kAlpacaStyleInstructionHandler",
    ]
    if config.prompt_type is not None and config.handler_name not in support_prompt_type_handler:
        raise ValueError(f'If specify prompt_type , handler name must be in:\n{support_prompt_type_handler}.')

    if (config.merge_group_keys is not None) and (not os.path.isdir(config.input)):
        raise ValueError(f"{config.input} is not a directory or does not exist")

    if not os.path.isdir(os.path.dirname(config.output_prefix)):
        raise ValueError(f"{os.path.dirname(config.output_prefix)} is not a directory or does not exist")

    if not config.pack and config.neat_pack:
        raise ValueError("Require set `pack` True when `neat-pack` is True.")

    if config.enable_thinking and config.prompt_type != 'qwen3':
        raise ValueError("enable_thinking only support when using qwen3 prompt type")
