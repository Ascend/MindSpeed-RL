# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.config_cls.base_config import BaseConfig


class RLConfig(BaseConfig):
    """RL configuration class for reinforcement learning training.

    This class manages comprehensive RL training parameters including model resources,
    training hyperparameters, dispatch configurations, and multimodal settings.
    All instance attributes are initialized using the dictionary keys.

    Attributes:
        runtime_env_path (str): Path to runtime environment configuration file.
        hccl_buffersize (int): HCCL buffer size.
        rule_reward (bool): Whether to use rule-based rewards in addition to model-based rewards.
        beta (float): Weight coefficient for balancing rule-based and model-based rewards.
        actor_resource (dict): Resource configuration for the actor model (e.g., GPU/CPU allocation).
        reference_resource (dict): Resource configuration for the reference model (e.g., GPU/CPU allocation).
        reward_resource (dict): Resource configuration for the reward model (e.g., GPU/CPU allocation).
        critic_resource (dict): Resource configuration for the critic model (e.g., GPU/CPU allocation).
        vit_resource (dict): Resource configuration for the visual model (e.g., GPU/CPU allocation).
        num_samples_per_step (int): Number of samples per step.
        max_prompt_length (int): Maximum prompt length.
        epochs (int): Number of epochs.
        clip_ratio (float): Clipping ratio.
        cliprange_value (float): Clipping range for value function.
        entropy_coeff (float): Coefficient for entropy regularization.
        gamma (float): Discount factor for future rewards (used in reinforcement learning).
        lam (float): Lambda parameter for Generalized Advantage Estimation (GAE).
        advantage_whiten (bool): Whether to normalize (whiten) advantages for stability.
        kl_penalty (str): Type of KL penalty to apply (e.g., 'kl', 'reverse_kl', 'low_var_kl').
        kl_ctrl_type (str): Type of KL divergence control (e.g., 'fixed', 'adaptive').
        init_kl_coef (float): Initial coefficient for KL divergence penalty.
        kl_horizon (int): Time horizon for KL divergence control (used in adaptive methods).
        kl_target (float): Target value for KL divergence (used in adaptive methods).
        adv_estimator (str): Method for estimating advantages (e.g., 'group_norm', 'gae').
        verifier_function (list): List of verifier functions to use.
        verifier_weight (list): List of weights for verifier functions.
        verifier_parallel (int): Parallel degree for verifier.
        verifier_timeout (int): Timeout for verifier in seconds.
        integrated_mode_config (dict): Configuration for integrated mode.
        use_kl_in_reward (bool): Whether to enable in-reward kl penalty.
        shuffle_mini_batch (bool): Whether to shuffle minibatch.
        tp_split_expert (bool): Whether to split experts across tensor parallel ranks.
        use_tensorboard (bool): Whether to use tensorboard.
        use_wandb (bool): Whether to use wandb.
        wandb_project (str): The wandb project name.
        wandb_exp_name (str): The wandb experiment name.
        wandb_save_dir (str): Path to save the wandb results locally.
        use_swanlab (bool): Whether to use swanlab.
        swanlab_mode (str): Swanlab mode.
        swanlab_project (str): Swanlab project name.
        swanlab_exp_name (str): Swanlab experiment name.
        swanlab_save_dir (str): Path to save the swanlab results locally.
        blocking (bool): Whether to enable blocking mode.
        async_engine (bool): Whether to enable the asynchronous generate process.
        guarantee_order (bool): Whether to guarantee order in async engine.
        num_cpus_for_local_task (int): Number of CPUs for local ray task.
        num_cpus_for_placement_group (int): Number of CPUs for ray worker placement group.
        use_integrated_worker (bool): Whether to use integrated worker.
        use_dp_batch_balance (bool): Whether to use dynamic batch size balancing across data parallel ranks.
        ref_forward_micro_batch_size (int): Micro batch size for ref log_p calculation.
        actor_forward_micro_batch_size (int): Micro batch size for actor log_p calculation.
        vit_forward_micro_batch_size (int): Micro batch size for vit calculation.
        actor_rollout_dispatch_size (int): Experience count every forward step for generate.
        actor_logprob_dispatch_size (int): Experience count every forward step for actor_logprob.
        ref_dispatch_size (int): Experience count every forward step for reference.
        reward_dispatch_size (int): Experience count every forward step for reward.
        dynamic_sampling_dispatch_size (int): Experience count every forward step for dynamic sampling.
        adv_dispatch_size (int): Experience count every forward step for advantages.
        actor_update_dispatch_size (int): Experience count every forward step for actor update.
        critic_update_dispatch_size (int): Experience count every forward step for critic update.
        critic_value_dispatch_size (int): Experience count every forward step for critic value.
        kl_dispatch_size (int): Experience count every forward step for kl divergence.
        actor_image_embeds_dispatch_size (int): Experience count every forward step for actor image embeds.
        is_multimodal (bool): Whether base model is a multimodal model or not.
        use_remove_padding (bool): Whether to use packed sequences for forward.
        reuse_image_embeds (bool): Whether to reuse image embeds for vit calculation.
        colocate_actor_and_vit (bool): Whether to colocate actor and vit models.
        n_samples_per_prompt (int): Number of samples per prompt.
        mini_batch_size (int): Mini batch size.
        use_dynamic_bsz (bool): Whether to use dynamic batch size.
        max_packing_token_size (int): Maximum token size for packing.
        ref_max_packing_token_size (int): Maximum token size for reference packing.
        actor_max_packing_token_size (int): Maximum token size for actor packing.
        update_max_packing_token_size (int): Maximum token size for update packing.
        dynamic_max_batch_size (int): Maximum batch size for dynamic batching.
        ref_dynamic_max_batch_size (int): Maximum batch size for reference dynamic batching.
        actor_dynamic_max_batch_size (int): Maximum batch size for actor dynamic batching.
        update_dynamic_max_batch_size (int): Maximum batch size for update dynamic batching.
        log_max_throughput (bool): Whether to log maximum throughput.
        token_level_loss (bool): Whether to use token level loss.
        data_strategy (str): Data strategy for training.
        transfer_queue_data_shard_num (int): Number of data shards for transfer queue.
        transfer_queue_data_shard_port_base (int): Base port for transfer queue data shards.
        clip_higher_enable (bool): Whether to enable clip higher strategy.
        clip_ratio_low (float): Lower clipping ratio for clip higher strategy.
        clip_ratio_high (float): Higher clipping ratio for clip higher strategy.
        overlong_buffer_enable (bool): Whether to enable overlong buffer penalty.
        overlong_buffer (int): Overlong buffer size.
        overlong_buffer_penalty_factor (float): Penalty factor for overlong buffer.
        rollout_max_tokens (int): Maximum tokens for rollout.
        filter_groups_enable (bool): Whether to enable filter groups.
        filter_groups_metric (str): Metric for filter groups.
        filter_groups_max_batches (int): Maximum batches for filter groups.
        filter_groups_train_batch_size (int): Train batch size for filter groups.
        zmq_communication (bool): Whether to use zmq for dispatch data.
        partial_rollout_max_split (int): The multiple of token splitting for max tokens when partial rollout is enabled.
        require_max_age_all_finished (bool): Whether to require the responses that have reached max_age must be completed in this iteration.
        multi_turn_enable (bool): Whether to enable multi-turn conversation.
        tool_config_path (str): Path to tool configuration file.
        max_tool_calls (int): Maximum number of tool calls.
        max_parallel_calls (int): Maximum number of parallel tool calls.
        max_total_response_length (int): Maximum total response length.
        max_tool_response_length (int): Maximum tool response length.
        tool_response_truncate_side (str): Side to truncate tool response.
        tool_parser_format (str): Format for tool parser.
        share_backbone (bool): Whether to share backbone parameters for LoRA.
    """

    def __init__(self, config_dict):
        """Initialize RLConfig with configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing the configuration parameters.
        """
        self.runtime_env_path = 'configs/envs/runtime_env.yaml'
        self.hccl_buffersize = 256
        self.rule_reward = True
        self.beta = 0.1
        self.actor_resource = {"num_npus": None}
        self.reference_resource = None
        self.reward_resource = None
        self.critic_resource = None
        self.vit_resource = None
        self.num_samples_per_step = 1
        self.max_prompt_length = 512
        self.epochs = 1
        self.clip_ratio = 0.2
        self.cliprange_value = 0.5
        self.entropy_coeff = 0.0
        self.gamma = 1.0
        self.lam = 0.95
        self.advantage_whiten = True
        self.kl_penalty = "low_var_kl"
        self.kl_ctrl_type = 'fixed'
        self.init_kl_coef = 0.01
        self.kl_horizon = 1000
        self.kl_target = 100.0
        self.adv_estimator = 'group_norm'
        self.verifier_function = ["base_acc", ]
        self.verifier_weight = [1.0, ]
        self.verifier_parallel = 1
        self.verifier_timeout = 30
        self.integrated_mode_config = None
        self.use_kl_in_reward = False

        self.shuffle_mini_batch = False
        self.tp_split_expert = False

        self.use_tensorboard = False
        self.use_wandb = False
        self.wandb_project = ""
        self.wandb_exp_name = ""
        self.wandb_save_dir = ""
        self.use_swanlab = False
        self.swanlab_mode = ""
        self.swanlab_project = ""
        self.swanlab_exp_name = ""
        self.swanlab_save_dir = ""
        self.blocking = True
        self.async_engine = False
        self.guarantee_order = False
        self.num_cpus_for_local_task = 1
        self.num_cpus_for_placement_group = 8
        self.use_integrated_worker = True
        self.use_dp_batch_balance = False
        self.ref_forward_micro_batch_size = None
        self.actor_forward_micro_batch_size = None
        self.vit_forward_micro_batch_size = None

        self.actor_rollout_dispatch_size = None
        self.actor_logprob_dispatch_size = None
        self.ref_dispatch_size = None
        self.reward_dispatch_size = None
        self.dynamic_sampling_dispatch_size = None
        self.adv_dispatch_size = None
        self.actor_update_dispatch_size = None
        self.critic_update_dispatch_size = None
        self.critic_value_dispatch_size = None
        self.kl_dispatch_size = None
        self.actor_image_embeds_dispatch_size = None

        self.is_multimodal = False
        self.use_remove_padding = False
        self.reuse_image_embeds = False
        self.colocate_actor_and_vit = False

        self.n_samples_per_prompt = config_dict.get('n_samples_per_prompt', 1)
        self.mini_batch_size = 1

        self.use_dynamic_bsz = False
        self.max_packing_token_size = 4096
        self.ref_max_packing_token_size = self.max_packing_token_size
        self.actor_max_packing_token_size = self.max_packing_token_size
        self.update_max_packing_token_size = self.max_packing_token_size
        self.dynamic_max_batch_size = None
        self.ref_dynamic_max_batch_size = self.dynamic_max_batch_size
        self.actor_dynamic_max_batch_size = self.dynamic_max_batch_size
        self.update_dynamic_max_batch_size = self.dynamic_max_batch_size

        self.log_max_throughput = True

        self.token_level_loss = True
        self.data_strategy = "td"
        self.transfer_queue_data_shard_num = config_dict.get('transfer_queue_data_shard_num', 1)
        self.transfer_queue_data_shard_port_base = config_dict.get('transfer_queue_data_shard_port_base', None)

        self.clip_higher_enable = False
        self.clip_ratio_low = 0.1
        self.clip_ratio_high = 0.1

        self.overlong_buffer_enable = False
        self.overlong_buffer = 0
        self.overlong_buffer_penalty_factor = 1.0
        self.rollout_max_tokens = 2048

        self.filter_groups_enable = False
        self.filter_groups_metric = "acc"
        self.filter_groups_max_batches = 1
        self.filter_groups_train_batch_size = 1

        self.zmq_communication = True
        self.partial_rollout_max_split = 1
        self.require_max_age_all_finished = True

        self.multi_turn_enable = False
        self.tool_config_path = None
        self.max_tool_calls = 0
        self.max_parallel_calls = 0
        self.max_total_response_length = 0
        self.max_tool_response_length = 0
        self.tool_response_truncate_side = None
        self.tool_parser_format = None

        self.share_backbone = False

        if config_dict.get("actor_resource") is not None:
            for key, _ in config_dict["actor_resource"].items():
                if key not in self.actor_resource:
                    raise ValueError(f"The key: {key} is missing, causing the setup to fail. Please check."
                            f" If necessary, register it in the config file.")

        self.update(config_dict)
        self.mini_batch_size = config_dict.get('mini_batch_size', 1) * self.n_samples_per_prompt