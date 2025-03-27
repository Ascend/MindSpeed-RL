from mindspeed_rl.config_cls.base_config import BaseConfig


class RLConfig(BaseConfig):
    '''
    RL configuration class.
    Initialize model configuration from the provided config dictionary.
    All instance attributes are initialized using the dictionary keys.

    param config_dict: Dictionary containing the configuration parameters
    rule_reward: Whether to use rule-based rewards in addition to model-based rewards (default: True)
    beta: Weight coefficient for balancing rule-based and model-based rewards (default: 0.1)
    actor_resource: Resource configuration for the actor model (e.g., GPU/CPU allocation) (default: None)
    reference_resource: Resource configuration for the reference model (e.g., GPU/CPU allocation) (default: None)
    reward_resource: Resource configuration for the reward model (e.g., GPU/CPU allocation) (default: None)
    mini_batch_size: Mini batch size (default: 1)
    num_samples_per_step: Number of samples per step (default: 1)
    max_prompt_length: Maximum prompt length (default: 512)
    epochs: Number of epochs (default: 1)
    clip_ratio: Clipping ratio (default: 0.2)
    entropy_coeff: Coefficient for entropy regularization (default: 0.001)
    gamma:  Discount factor for future rewards (used in reinforcement learning) (default: 1.0)
    lam: Lambda parameter for Generalized Advantage Estimation (GAE) (default: 0.95)
    advantage_whiten: Whether to normalize (whiten) advantages for stability (default: True)
    kl_ctrl_type: Type of KL divergence control (e.g., 'fixed', 'adaptive') (default: 'fixed')
    init_kl_coef: Initial coefficient for KL divergence penalty (default: 0.01)
    kl_horizon: Time horizon for KL divergence control (used in adaptive methods) (default: 1000)
    kl_target: Target value for KL divergence (used in adaptive methods) (default: 100.0)
    adv_estimator: Method for estimating advantages (e.g., 'group_norm', 'gae') (default: 'group_norm')
    kl_penalty: Type of KL penalty to apply (e.g., 'kl', 'reverse_kl') (default: 'kl')

    experience_count: The batch size of prompts per pipeline stage in each data fetch operation from the TransferDock (TD). (default: 1)
    experience_count_actor: experience count every forward step for actor (default: same as experience_count_all)
    experience_count_reward: experience count every forward step for reward (default: same as experience_count_all)
    experience_count_ref: experience count every forward step for reference (default: same as experience_count_all)
    experience_count_rule_reward: experience count every forward step for rule reward (default: same as experience_count_all)

    shuffle_mini_batch: Whether to shuffle minibatch (default: False)
    n_samples_per_prompt: Number of samples per prompt (default: 1)
    enable_sharding_validate: Whether to enable sharding validation (default: False)
    tp_split_expert:
    use_tensorboard: Whether to use tensorboard (default: False)
    use_wandb: Whether to use wandb (default: False)
    wandb_project: The wandb project name. Ignore wandb by default. If use_wandb is True, you need to set the wandb project name (default: "")
    wandb_exp_name: The wandb experiment name. If use_wandb is True, you need to set the wandb experiment name (default: "")
    wandb_save_dir: Path to save the wandb results locally. (default: "")
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
        self.gamma = 1.0
        self.lam = 0.95
        self.advantage_whiten = True
        self.kl_ctrl_type = 'fixed'
        self.init_kl_coef = 0.01
        self.kl_horizon = 1000
        self.kl_target = 100.0
        self.adv_estimator = 'group_norm'
        self.kl_penalty = 'kl'
        self.verifier_function = ["base_acc", ]
        self.verifier_weight = [1.0, ]
        self.verifier_parallel = 1
        self.verifier_timeout = 30

        self.experience_count = 1
        self.experience_count_actor = None
        self.experience_count_reward = None
        self.experience_count_ref = None
        self.experience_count_rule_reward = None

        self.shuffle_mini_batch = False
        self.n_samples_per_prompt = 1
        self.enable_sharding_validate = False
        self.tp_split_expert = False

        self.use_tensorboard = False
        self.use_wandb = False
        self.wandb_project = ""
        self.wandb_exp_name = ""
        self.wandb_save_dir = ""
        self.blocking = False
        self.num_cpus_for_local_task = 1

        self.update(config_dict)
