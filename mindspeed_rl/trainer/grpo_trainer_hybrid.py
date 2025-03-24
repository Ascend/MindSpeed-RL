# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import copy
from typing import List, Union
import ray
import torch
from codetiming import Timer
from torch.utils.data import DataLoader

from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.trainer.base import RayBaseTrainer
from mindspeed_rl.trainer.utils.transfer_dock import GRPOTransferDock
from mindspeed_rl.trainer.utils.compute_utils import compute_advantage, compute_grpo_data_metrics
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.utils import metrics_post_processing, compute_tps


class RayGRPOTrainer(RayBaseTrainer):
    """
    RayGRPOTrainer class. This trainer runs on the driver process on a single CPU/GPU node.

    Args:
        actor_worker: RayActorGroup The actor worker group.
        ref_worker: RayActorGroup The reference worker group.
        reward_list: List[Union[RayActorGroup, RuleReward]] List of reward workers or rule-based rewards.
        train_iters: int = 1 The number of training iterations.
        save_interval: int = 1 The interval (in iterations) for saving checkpoints.
        kl_ctrl_type: str = 'fixed' The type of KL divergence control (e.g., 'fixed', 'adaptive').
        adv_estimator: str = "group_norm" The method for estimating advantages (e.g., 'group_norm', 'mean').
        kl_horizon: int = 1000 The time horizon for KL divergence control (used in adaptive methods).
        kl_target: float = 100.0 The target value for KL divergence (used in adaptive methods).
        init_kl_coef: float = 0.01 The initial coefficient for KL divergence penalty.
        global_batch_size: int = 32 The global batch size for training.
        micro_batch_size: int = 1 The micro batch size for gradient accumulation.
        n_samples_per_prompt: int = 1 The number of samples generated per prompt.
        tokenizer_name_or_path: str = None The name or path of the tokenizer to use.
        dataset_additional_keys: List[str] = None Additional keys to include in the dataset.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            actor_worker: RayActorGroup,
            ref_worker: RayActorGroup,
            reward_list: List[Union[RayActorGroup, RuleReward]],
            train_iters: int = 1,
            save_interval: int = 1,
            kl_ctrl_type: str = 'fixed',
            adv_estimator: str = "group_norm",
            kl_horizon: int = 1000,
            kl_target: float = 100.0,
            init_kl_coef: float = 0.01,
            global_batch_size: int = 32,
            micro_batch_size: int = 1,
            n_samples_per_prompt: int = 1,
            tokenizer_name_or_path: str = None,
            dataset_additional_keys: List[str] = None,
            blocking: bool = False,
            num_cpus_for_local_task: float = 0.1,
            **kwargs
    ):
        super().__init__(
            actor_worker,
            ref_worker,
            reward_list,
            train_iters=train_iters,
            save_interval=save_interval,
            kl_ctrl_type=kl_ctrl_type,
            kl_horizon=kl_horizon,
            kl_target=kl_target,
            adv_estimator=adv_estimator,
            init_kl_coef=init_kl_coef,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dataset_additional_keys=dataset_additional_keys,
            blocking=blocking,
            num_cpus_for_local_task=num_cpus_for_local_task,
            **kwargs
        )

        self.transfer_dock = None
        self.metrics = Metric()
        self.transfer_dock_init()
        self.kwargs = kwargs

    def transfer_dock_init(self):
        self.transfer_dock = GRPOTransferDock.remote(self.global_batch_size, self.metrics, addition_columns=self.dataset_additional_keys)
        self.actor_worker.sync_init_transfer_dock(self.transfer_dock)
        self.ref_worker.sync_init_transfer_dock(self.transfer_dock)
        for reward in self.reward_list:
            if hasattr(reward, 'sync_init_transfer_dock'):
                reward.sync_init_transfer_dock(self.transfer_dock)
            else:
                reward.init_transfer_dock.remote(self.transfer_dock)

    def fit(self, data_loader: DataLoader):
        """
        The utils loop of GRPO
        """
        logger = Loggers('grpo_trainer_hybrid')
        metrics = Metric()

        data_iters = iter(data_loader)

        iteration = self.actor_worker.get_iteration()

        while iteration < self.train_iters:

            batch = next(data_iters)
            prompts = batch['prompts']
            prompt_length = []
            for prompt in prompts:
                for _ in range(self.n_samples_per_prompt):
                    prompt_length.append(torch.tensor([len(prompt)]))

            prompts_data = prompts
            prompts = []
            for prompt in prompts_data:
                for _ in range(self.n_samples_per_prompt):
                    prompts.append(copy.deepcopy(prompt))

            add_vals = {}
            for add_keys in self.dataset_additional_keys:
                if add_keys in batch.keys():
                    values = []
                    for value in batch[add_keys]:
                        for _ in range(self.n_samples_per_prompt):
                            values.append(value)
                    add_vals[add_keys] = values

            ray.get(self.transfer_dock.clear.remote())
            ray.get(self.transfer_dock.put_experience.remote(
                    data_dict=dict({'prompt_length': prompt_length, 'prompts': prompts}, **add_vals),
                    num_responses=self.n_samples_per_prompt))

            with Timer(name='iteration', logger=None) as all_timer:
                # generate sequences
                self.actor_worker.generate_sequences(blocking=self.blocking)

                # compute reference log_prob
                self.ref_worker.compute_log_prob(blocking=self.blocking)

                # compute rm scores.
                for reward_worker in self.reward_list:
                    if isinstance(reward_worker, RayActorGroup):
                        reward_worker.compute_rm_score(blocking=self.blocking)
                    else:
                        self.rule_reward_compute_rm_score(reward_worker, blocking=self.blocking)

                # compute advantages, executed on the driver process
                self.compute_advantage(blocking=self.blocking)

                self.actor_worker.wait_all_ref_objs_run_over()
                # compute old log_prob
                self.actor_worker.compute_log_prob(blocking=self.blocking)

                # update actor
                self.actor_worker.update(self.kl_ctrl)

                # collect metrics
                grpo_data_metrics = compute_grpo_data_metrics(self.transfer_dock,
                                                              self.global_batch_size,
                                                              self.tokenizer_name_or_path)
                metrics_result = ray.get(self.transfer_dock.get_metrics.remote())

            metrics_result = metrics_post_processing(metrics_result)
            tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, all_timer.last)

            metrics.update(value=grpo_data_metrics)
            metrics.update(value=metrics_result)
            metrics.update("timing/all", all_timer.last)
            metrics.update("tokens/p/s", tps)
            iteration += 1
            logger.info(metrics.metric, iteration, self.train_iters)
            if self.tensorboard is not None:
                for k, v in metrics.metric.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iteration)
            if self.wandb is not None:
                self.wandb.log_metrics(metrics.metric, iteration)
            if iteration % self.save_interval == 0:
                self.save_checkpoint(iteration)

    def compute_advantage(self, blocking=False):
        compute_advantage_ref = compute_advantage.options(num_cpus=self.num_cpus_for_local_task).remote(
            self.transfer_dock,
            self.gamma,
            self.lam,
            adv_estimator=self.adv_estimator,
            experience_count=self.micro_batch_size,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            n_samples_per_prompt=self.n_samples_per_prompt,
        )
        if blocking:
            ray.get(compute_advantage_ref)

    @staticmethod
    def rule_reward_compute_rm_score(reward_worker, blocking=False):
        rule_reward_compute_rm_score_ref = reward_worker.compute_rm_score.remote()
        if blocking:
            ray.get(rule_reward_compute_rm_score_ref)

    def save_checkpoint(self, iteration: int):
        self.actor_worker.save_checkpoint(iteration)
