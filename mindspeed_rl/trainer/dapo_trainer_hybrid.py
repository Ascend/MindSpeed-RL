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
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.trainer.utils.transfer_dock import GRPOTransferDock
from mindspeed_rl.trainer.utils.compute_utils import compute_advantage, compute_dapo_data_metrics, dynamic_sampling
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.utils import metrics_post_processing, compute_tps, compute_vllm_throughput, metrics_sort


class RayDAPOTrainer(RayBaseTrainer):
    """
    RayDAPOTrainer class. This trainer runs on the driver process on a single CPU/GPU node.

    Args:
        actor_worker: RayActorGroup The actor worker group.
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
            guarantee_order: bool = False,
            num_cpus_for_local_task: int = 1,
            **kwargs
    ):
        super().__init__(
            actor_worker,
            None,
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
            guarantee_order=guarantee_order,
            num_cpus_for_local_task=num_cpus_for_local_task,
            **kwargs
        )

        self.transfer_dock = None
        self.sampling_transfer_dock = None
        self.metrics = Metric()
        self.kwargs = kwargs
        self.should_filter = self.kwargs['filter_groups_enable']
        self.max_num_prompt_in_batch = self.kwargs['filter_groups_train_batch_size']
        self.max_num_gen_batches = self.kwargs['filter_groups_max_batches']
        self.transfer_dock_init()
        self.num_prompt_in_batch = 0
        self.num_gen_batches = 0
        self.set_actor_log_prob_skip_flag()

    def transfer_dock_init(self):
        if self.should_filter:
            # 存储动态采样过滤后的数据，用于计算损失、更新权重等
            self.transfer_dock = GRPOTransferDock.remote(self.max_num_prompt_in_batch, self.n_samples_per_prompt,
                                                         self.metrics, addition_columns=self.dataset_additional_keys)
            # 存储动态采样过滤前的数据，用于生成response、计算奖励等
            self.sampling_transfer_dock = GRPOTransferDock.remote(self.global_batch_size,
                                                              self.n_samples_per_prompt,
                                                              self.metrics,
                                                              addition_columns=self.dataset_additional_keys)
            self.actor_worker.sync_init_transfer_dock(self.transfer_dock, sampling_transfer_dock=self.sampling_transfer_dock)
            if self.ref_worker:
                self.ref_worker.sync_init_transfer_dock(self.transfer_dock, sampling_transfer_dock=self.sampling_transfer_dock)
            for reward in self.reward_list:
                if hasattr(reward, 'sync_init_transfer_dock'):
                    reward.sync_init_transfer_dock(self.transfer_dock, sampling_transfer_dock=self.sampling_transfer_dock)
                else:
                    reward.init_transfer_dock.remote(self.transfer_dock, sampling_transfer_dock=self.sampling_transfer_dock)
        else:
            self.transfer_dock = GRPOTransferDock.remote(self.global_batch_size, self.n_samples_per_prompt,
                                                         self.metrics, addition_columns=self.dataset_additional_keys)
            self.actor_worker.sync_init_transfer_dock(self.transfer_dock)
            if self.ref_worker:
                self.ref_worker.sync_init_transfer_dock(self.transfer_dock)
            for reward in self.reward_list:
                if hasattr(reward, 'sync_init_transfer_dock'):
                    reward.sync_init_transfer_dock(self.transfer_dock)
                else:
                    reward.init_transfer_dock.remote(self.transfer_dock)

    def set_actor_log_prob_skip_flag(self):
        global_batch_size = self.actor_worker.megatron_config.global_batch_size
        mini_batch_size = self.actor_worker.rl_config.mini_batch_size
        n_samples_per_prompt = self.actor_worker.rl_config.n_samples_per_prompt
        epochs = self.actor_worker.rl_config.epochs
        self.skip_actor_log_prob = (global_batch_size * n_samples_per_prompt == mini_batch_size and epochs == 1)
        self.actor_worker.skip_actor_log_prob = self.skip_actor_log_prob

    def fit(self, data_iters):
        """
        The utils loop of DAPO
        """
        logger = Loggers('dapo_trainer_hybrid')
        metrics = Metric()

        iteration = self.actor_worker.get_iteration()

        if self.blocking:
            logger.info('sync start dapo training at iteration: {}/{} ...'.format(iteration, self.train_iters))
        else:
            logger.info('async start dapo training at iteration: {}/{} ...'.format(iteration, self.train_iters))

        all_time = 0
        rollout_time = 0
        rule_reward_time = 0
        while iteration < self.train_iters:
            start_time = time.time()
            batch = next(data_iters)
            data_num = self.global_batch_size * self.n_samples_per_prompt
            if self.should_filter:
                ray.get(self.sampling_transfer_dock.clear.remote())
                index_list = ray.get(self.sampling_transfer_dock.prefetch_request_index.remote(data_num))
                if index_list:
                    ray.get(self.sampling_transfer_dock.put_prompts_experience.remote(batch, self.dataset_additional_keys, index_list))
            else:
                ray.get(self.transfer_dock.clear.remote())
                ray.get(self.transfer_dock.put_prompts_experience.remote(batch, self.dataset_additional_keys))
            
            # generate sequences
            logger.info(f"dapo fit generate_sequences")
            if self.should_filter:
                self.actor_worker.enter_infer_mode()
            rollout_start_time = time.time()
            self.actor_worker.generate_sequences(blocking=self.blocking)
            # compute rm scores.
            logger.info(f"dapo fit compute_rm_score")
            rule_reward = []
            rule_reward_start_time = time.time()
            for reward_worker in self.reward_list:
                if isinstance(reward_worker, RayActorGroup):
                    reward_worker.compute_rm_score(blocking=self.blocking)
                else:
                    rule_reward.append(reward_worker.compute_rm_score.remote())
            ray.get(rule_reward)
            rule_reward_end_time = time.time()
            rule_reward_time += rule_reward_end_time - rule_reward_start_time
            rollout_time += rule_reward_end_time - rollout_start_time

            if self.should_filter:
                # dynamic sampling
                logger.info(f"dapo fit dynamic_sampling")
                should_continue = self.dynamic_sampling(data_num)
                if should_continue:
                    end_time = time.time()
                    all_time += end_time - start_time
                    continue
                ray.get(self.sampling_transfer_dock.clear.remote())
                self.actor_worker.exit_infer_mode()
                data_num = self.max_num_prompt_in_batch * self.n_samples_per_prompt

            logger.info(f"dapo fit compute_advantage")
            # compute advantages, executed on the driver process
            self.compute_advantage(data_num, blocking=False, guarantee_order=self.guarantee_order)

            logger.info(f"dapo fit compute_log_prob {self.skip_actor_log_prob}")
            # compute old log_prob
            if not self.skip_actor_log_prob:
                self.actor_worker.compute_log_prob(blocking=self.blocking)

            self.actor_worker.wait_all_ref_objs_run_over()

            for reward in self.reward_list:
                if hasattr(reward, 'wait_all_ref_objs_run_over'):
                    reward.wait_all_ref_objs_run_over()

            logger.info(f"dapo fit update")
            # update actor
            self.actor_worker.update(self.kl_ctrl, self.skip_actor_log_prob)

            logger.info(f"dapo fit compute_dapo_data_metrics")
            # collect metrics

            dapo_data_metrics = compute_dapo_data_metrics(self.transfer_dock,
                                                          data_num,
                                                          self.tokenizer,
                                                          data_num,
                                                          self.guarantee_order)
            metrics_result = ray.get(self.transfer_dock.get_metrics.remote())
            end_time = time.time()
            all_time += end_time - start_time
            sharding_infer_intervals = metrics_result.metric.get("timing/resharding_to_infer", [])
            sharding_infer_interval = metrics_result.compute_sum(sharding_infer_intervals)
            if self.should_filter:
                filter_intervals = metrics_result.metric.get("timing/filter", [])
                filter_interval = metrics_result.compute_sum(filter_intervals)

            metrics_result = metrics_post_processing(metrics_result)
            metrics_result = metrics_sort(metrics_result, all_time)
            tps = compute_tps(self.kwargs, dapo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                              all_time)
            vllm_throughput = compute_vllm_throughput(self.kwargs, dapo_data_metrics, self.global_batch_size,
                                                      self.n_samples_per_prompt, metrics_result["timing/rollout"])
            metrics.update(value=metrics_result)
            metrics.update(value=dapo_data_metrics)
            metrics.update("tokens/p/s", tps)
            metrics.update("vllm_throughput", vllm_throughput)
            metrics.update("train/num_gen_batches", self.num_gen_batches)
            metrics.update("timing/resharding_to_infer", sharding_infer_interval)
            metrics.update("timing/rule_reward", rule_reward_time)
            metrics.update("timing/rollout", rollout_time - rule_reward_time)
            if self.should_filter:
                metrics.update("timing/filter", filter_interval)
            metrics.remove_key("timing/non_overlap_reference_model")
            metrics.remove_key("timing/non_overlap_rule_reward")
            metrics.remove_key("timing/non_overlap_adv")
            metrics.remove_key("actor/kl_loss")
            metrics.remove_grpo_keys()

            iteration += 1
            all_time = 0
            rollout_time = 0
            rule_reward_time = 0
            logger.info(metrics.metric, iteration, self.train_iters)
            if self.tensorboard is not None:
                for k, v in metrics.metric.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iteration)
            if self.wandb is not None:
                self.wandb.log_metrics(metrics.metric, iteration)
            if iteration % self.save_interval == 0 or iteration == self.train_iters:
                self.save_checkpoint(iteration)
            
            if self.should_filter:
                logger.info(f"dapo fit clear")
                ray.get(self.transfer_dock.clear.remote())
                # 刷新current train prompt num, 和filter_groups_train_batch_size比较
                self.num_prompt_in_batch = 0
                # 刷新current gen batch num, 和filter_groups_max_batches比较
                self.num_gen_batches = 0

        logger.info('after dapo training is done')
        ray.shutdown()

    def dynamic_sampling(self, data_num):
        start_filter_time = time.time()
        self.num_gen_batches += 1
        dynamic_sampling_ref = dynamic_sampling.options(num_cpus=self.num_cpus_for_local_task).remote(
            self.num_prompt_in_batch,
            data_num,
            self.n_samples_per_prompt,
            self.sampling_transfer_dock,
            self.transfer_dock,
            self.guarantee_order
        )

        self.num_prompt_in_batch = ray.get(dynamic_sampling_ref)
        end_filter_time = time.time()
        ray.get(
            self.transfer_dock.update_metrics.remote(
                "timing/filter",
                value=[end_filter_time - start_filter_time],
                cumulate=True
            )
        )

        if self.num_prompt_in_batch < self.max_num_prompt_in_batch:
            if self.max_num_gen_batches <= 0 or self.num_gen_batches < self.max_num_gen_batches:
                return True
            else:
                raise ValueError('Generated too many. Please check your data.')

        return False

    def compute_advantage(self, data_num, blocking=False, guarantee_order=False):
        experience_count = self.micro_batch_size

        start_adv_time = time.time()
        compute_advantage_ref = compute_advantage.options(num_cpus=self.num_cpus_for_local_task).remote(
            self.transfer_dock,
            self.gamma,
            self.lam,
            adv_estimator=self.adv_estimator,
            experience_count=experience_count,
            tokenizer=self.tokenizer,
            global_batch_size=data_num,
            guarantee_order=guarantee_order,
            n_sample_per_prompt=self.n_samples_per_prompt
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