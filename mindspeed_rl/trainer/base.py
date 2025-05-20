# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import List, Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.trainer.utils.compute_utils import FixedKLController, AdaptiveKLController
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from mindspeed_rl.utils.loggers import WandbLogger


class RayBaseTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 actor_worker: RayActorGroup = None,
                 ref_worker: RayActorGroup = None,
                 reward_list: List[Union[RayActorGroup, RuleReward]] = None,
                 train_iters: int = 1,
                 save_interval: int = 1,
                 gamma: float = 1.0,
                 lam: float = 0.95,
                 adv_estimator: str = "group_norm",
                 missing_eos_penalty: float = 1.0,
                 kl_penalty: str = 'low_var_kl',
                 kl_ctrl_type: str = 'fixed',
                 kl_horizon: int = 1000,
                 kl_target: float = 100.0,
                 init_kl_coef: float = 0.001,
                 global_batch_size: int = 32,
                 micro_batch_size: int = 1,
                 n_samples_per_prompt: int = 1,
                 tokenizer: BaseTokenizer = None,
                 dataset_additional_keys: List[str] = None,
                 blocking: bool = False,
                 guarantee_order: bool = False,
                 num_cpus_for_local_task: float = 0.1,
                 **kwargs):

        self.actor_worker = actor_worker
        self.ref_worker = ref_worker
        self.reward_list = reward_list
        self.train_iters = train_iters
        self.save_interval = save_interval
        self.gamma = gamma
        self.lam = lam
        self.adv_estimator = adv_estimator
        self.missing_eos_penalty = missing_eos_penalty
        self.kl_penalty = kl_penalty
        self.kl_ctrl_type = kl_ctrl_type
        self.kl_horizon = kl_horizon
        self.kl_target = kl_target
        self.init_kl_coef = init_kl_coef
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.n_samples_per_prompt = n_samples_per_prompt
        self.tokenizer = tokenizer
        self.dataset_additional_keys = dataset_additional_keys
        self.blocking = blocking
        self.guarantee_order = guarantee_order
        self.num_cpus_for_local_task = num_cpus_for_local_task
        self.kwargs = kwargs

        # define KL control
        if kl_ctrl_type == 'fixed':
            self.kl_ctrl = FixedKLController(init_kl_coef=self.init_kl_coef)
        elif kl_ctrl_type == 'adaptive':
            if self.kl_horizon <= 0:
                raise ValueError(f'horizon must be larger than 0. Got {self.kl_horizon}')
            self.kl_ctrl = AdaptiveKLController(init_kl_coef=init_kl_coef,
                                                target_kl=kl_target,
                                                horizon=kl_horizon)
        else:
            raise NotImplementedError

        self.wandb = None
        self.tensorboard = None
        if kwargs.get("use_wandb", ""):
            self.wandb = WandbLogger(kwargs)
        if kwargs.get("use_tensorboard", "") and self.wandb is None:
            self.tensorboard = SummaryWriter()

    def transfer_dock_init(self):
        pass

    def fit(self, data_loader: DataLoader):
        """
        The utils loop of xx
        """
        pass

    def save_checkpoint(self, iteration):
        pass
