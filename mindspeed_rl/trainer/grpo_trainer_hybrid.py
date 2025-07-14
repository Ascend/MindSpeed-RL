# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import copy
import time
from typing import List, Union
import time
import ray
import torch
from codetiming import Timer
from torch.utils.data import DataLoader

from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.trainer.base import RayBaseTrainer
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig
from mindspeed_rl.trainer.utils import GRPOTransferDock, MMGRPOTransferDock
from mindspeed_rl.trainer.utils.compute_utils import compute_advantage, compute_grpo_data_metrics
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.utils import metrics_post_processing, compute_tps, metrics_sort, is_multimodal


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
        global_batch_size: int = 1 The global batch size for training (number of prompts per iteration).
        n_samples_per_prompt: int = 1 The number of samples generated per prompt.
        tokenizer: BaseTokenizer = None tokenizer to use.
        dataset_additional_keys: List[str] = None Additional keys to include in the dataset.
        blocking: bool = False  Whether to enable blocking mode.
        num_cpus_for_local_task: int = 1 Number of CPUs for local ray task.
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
            global_batch_size: int = 1,
            micro_batch_size: int = 1,
            n_samples_per_prompt: int = 1,
            tokenizer: BaseTokenizer = None,
            dataset_additional_keys: List[str] = None,
            blocking: bool = False,
            td_status_log: bool = False,
            guarantee_order: bool = False,
            num_cpus_for_local_task: int = 1,
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
            tokenizer=tokenizer,
            dataset_additional_keys=dataset_additional_keys,
            blocking=blocking,
            td_status_log=td_status_log,
            guarantee_order=guarantee_order,
            num_cpus_for_local_task=num_cpus_for_local_task,
            **kwargs
        )

        self.transfer_dock = None
        self.mm_transfer_dock = None
        self.metrics = Metric()
        self.transfer_dock_init()
        self.kwargs = kwargs
        self.set_actor_log_prob_skip_flag()

    def transfer_dock_init(self):
        self.transfer_dock = GRPOTransferDock.remote(self.global_batch_size, self.n_samples_per_prompt,
                                                     self.metrics, addition_columns=self.dataset_additional_keys, td_status_log=self.td_status_log)
        if is_multimodal():
            self.mm_transfer_dock = MMGRPOTransferDock.remote(self.global_batch_size, self.n_samples_per_prompt, td_status_log=self.td_status_log)

        self.actor_worker.sync_init_transfer_dock(self.transfer_dock, self.mm_transfer_dock)
        self.ref_worker.sync_init_transfer_dock(self.transfer_dock, self.mm_transfer_dock)
        for reward in self.reward_list:
            if hasattr(reward, 'sync_init_transfer_dock'):
                reward.sync_init_transfer_dock(self.transfer_dock, self.mm_transfer_dock)
            else:
                reward.init_transfer_dock.remote(self.transfer_dock, self.mm_transfer_dock)

    def set_actor_log_prob_skip_flag(self):
        global_batch_size = self.actor_worker.megatron_config.global_batch_size
        mini_batch_size = self.actor_worker.rl_config.mini_batch_size
        n_samples_per_prompt = self.actor_worker.rl_config.n_samples_per_prompt
        epochs = self.actor_worker.rl_config.epochs
        self.skip_actor_log_prob = (global_batch_size * n_samples_per_prompt == mini_batch_size and epochs == 1)
        self.actor_worker.skip_actor_log_prob = self.skip_actor_log_prob

    def fit(self, data_iters):
        """
        The utils loop of GRPO
        """
        logger = Loggers('grpo_trainer_hybrid')
        metrics = Metric()

        iteration = self.actor_worker.get_iteration()

        if self.blocking:
            logger.info('sync start grpo training at iteration: {}/{} ...'.format(iteration, self.train_iters))
        else:
            logger.info('async start grpo training at iteration: {}/{} ...'.format(iteration, self.train_iters))

        while iteration < self.train_iters:
            ray.get(self.transfer_dock.clear.remote())

            batch = next(data_iters)
            ray.get(self.transfer_dock.put_prompts_experience.remote(batch, self.dataset_additional_keys))
            if is_multimodal():
                ray.get(self.mm_transfer_dock.clear.remote())
                ray.get(self.mm_transfer_dock.put_experience.remote(batch, indexes=[i for i in range(len(batch['prompts']) * self.n_samples_per_prompt)], stage_tag="put_prompts_experience"))

            with Timer(name='iteration', logger=None) as all_timer:
                # generate sequences
                self.actor_worker.generate_sequences(blocking=self.blocking)

                # compute rm scores.
                rule_reward = []
                for reward_worker in self.reward_list:
                    if isinstance(reward_worker, RayActorGroup):
                        reward_worker.compute_rm_score(blocking=self.blocking)
                    else:
                        rule_reward.append(reward_worker.compute_rm_score.remote())

                # compute advantages, executed on the driver process
                self.compute_advantage(blocking=False, guarantee_order=self.guarantee_order)

                # compute reference log_prob
                self.ref_worker.compute_ref_log_prob(blocking=self.blocking)

                # compute old log_prob
                if not self.skip_actor_log_prob:
                    self.actor_worker.compute_log_prob(blocking=self.blocking)

                self.actor_worker.wait_all_ref_objs_run_over()

                self.ref_worker.wait_all_ref_objs_run_over()
                for reward in self.reward_list:
                    if hasattr(reward, 'wait_all_ref_objs_run_over'):
                        reward.wait_all_ref_objs_run_over()

                # update actor
                self.actor_worker.update(self.kl_ctrl, self.skip_actor_log_prob)

                # collect metrics
                grpo_data_metrics = compute_grpo_data_metrics(self.transfer_dock,
                                                              self.global_batch_size * self.n_samples_per_prompt,
                                                              self.tokenizer,
                                                              self.global_batch_size * self.n_samples_per_prompt,
                                                              self.guarantee_order)
                metrics_result = ray.get(self.transfer_dock.get_metrics.remote())

            metrics_result = metrics_post_processing(metrics_result)
            metrics_result = metrics_sort(metrics_result, all_timer.last)
            log_max_throughput = self.actor_worker.rl_config.log_max_throughput
            tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                              all_timer.last, log_max_throughput)
            update_tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                                     metrics_result["timing/update"], log_max_throughput)
            vllm_tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                                   metrics_result["timing/rollout"], log_max_throughput)
            metrics.update(value=metrics_result)
            metrics.update(value=grpo_data_metrics)
            metrics.update("e2e_tps", tps)
            metrics.update("update_tps", update_tps)
            metrics.update("vllm_tps", vllm_tps)
            iteration += 1
            logger.info(metrics.metric, iteration, self.train_iters)
            if self.tensorboard is not None:
                for k, v in metrics.metric.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iteration)
            if self.wandb is not None:
                self.wandb.log_metrics(metrics.metric, iteration)
            if iteration % self.save_interval == 0 or iteration == self.train_iters:
                self.save_checkpoint(iteration)

        logger.info('after grpo training is done')
        ray.shutdown()

    def compute_advantage(self, blocking=False, guarantee_order=False):
        experience_count = self.micro_batch_size

        start_adv_time = time.time()
        compute_advantage_ref = compute_advantage.options(num_cpus=self.num_cpus_for_local_task).remote(
            self.transfer_dock,
            self.gamma,
            self.lam,
            adv_estimator=self.adv_estimator,
            experience_count=experience_count,
            tokenizer=self.tokenizer,
            global_batch_size=self.global_batch_size * self.n_samples_per_prompt,
            guarantee_order=guarantee_order
        )
        if blocking:
            ray.get(compute_advantage_ref)
        end_adv_time = time.time()
        ray.get(
            self.transfer_dock.update_metrics.remote(
                "timing/adv",
                value=[round(end_adv_time, 4), round(start_adv_time, 4)],
                cumulate=True
            )
        )
        ray.get(
            self.transfer_dock.update_metrics.remote(
                "end_time/end_adv_time",
                value=[round(end_adv_time, 4)],
                cumulate=True
            )
        )

    def save_checkpoint(self, iteration: int):
        self.actor_worker.save_checkpoint(iteration)
