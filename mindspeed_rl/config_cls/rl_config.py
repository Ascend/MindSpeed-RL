# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.config_cls.base_config import BaseConfig


class RLConfig(BaseConfig):
    '''
    RL configuration class.
    Initialize model configuration from the provided config dictionary.
    All instance attributes are initialized using the dictionary keys.

    param config_dict: Dictionary containing the configuration parameters
    mini_batch_size: Mini batch size (default: 1)
    num_samples_per_step: Number of samples per step (default: 1)
    max_prompt_length: Maximum prompt length (default: 512)
    epochs: Number of epochs (default: 1)
    clip_ratio: Clipping ratio (default: 0.2)
    entropy_coeff: Coefficient for entropy regularization (default: 0.001)
    shuffle_mini_batch: Whether to shuffle minibatch (default: False)
    n_samples_per_prompt: Number of samples per prompt (default: 1)
    enable_sharding_validate: Whether to enable sharding validation (default: False)
    colocate_all_models: Whether to colocate all models (default: False)
    colocate_actor_ref: Whether to colocate actor and reference (default: False)
    # Default values can still be defined if no config is provided
    '''

    def __init__(self, config_dict):
        self.rule_reward = True
        self.beta = 0.1
        self.actor_resource = None
        self.reference_resource = None
        self.reward_resource = None
        self.mini_batch_size = 1
        self.num_samples_per_step = 1
        self.max_prompt_length = 512
        self.epochs = 1
        self.clip_ratio = 0.2
        self.entropy_coeff = 0.001
        self.shuffle_mini_batch = False
        self.n_samples_per_prompt = 1
        self.enable_sharding_validate = False
        self.tp_split_expert = False
        self.colocate_all_models = False
        self.colocate_actor_ref = False
        self.update(config_dict)
