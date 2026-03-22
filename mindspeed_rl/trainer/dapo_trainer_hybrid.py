# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import List, Union
import time
import ray

from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.dynamic_sampling import DynamicSampling
from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.trainer.base import RayBaseTrainer
from mindspeed_rl.trainer.utils.data_strategy import DataStrategy
from mindspeed_rl.trainer.utils.transfer_dock import put_prompts_experience
from mindspeed_rl.trainer.utils.compute_utils import compute_advantage, compute_dapo_data_metrics
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.utils import metrics_post_processing, compute_tps, metrics_sort, is_multimodal


class RayDAPOTrainer(RayBaseTrainer):
    """
    RayDAPOTrainer class for distributed DAPO (Dynamic Adaptive Policy Optimization) training.
    
    This trainer runs on the driver process on a single CPU/GPU node, coordinating
    distributed workers for actor model inference, reward computation, and dynamic sampling.
    """

    def __init__(
            self,
            actor_worker: RayActorGroup,
            reward_list: List[Union[RayActorGroup, RuleReward]],
            dynamic_sampling_list: List[DynamicSampling],
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
            partial_rollout_max_split: int = 1,
            **kwargs
    ):
        """
        Initialize the RayDAPOTrainer.
        
        Args:
            actor_worker: The actor worker group for policy generation.
            reward_list: List of reward workers or rule-based reward functions.
            dynamic_sampling_list: List of dynamic sampling controllers for data filtering.
            train_iters: The number of training iterations.
            save_interval: The interval (in iterations) for saving checkpoints.
            kl_ctrl_type: The type of KL divergence control ('fixed' or 'adaptive').
            adv_estimator: The method for estimating advantages (e.g., 'group_norm').
            kl_horizon: The time horizon for KL divergence control in adaptive mode.
            kl_target: The target value for KL divergence in adaptive mode.
            init_kl_coef: The initial coefficient for KL divergence penalty.
            global_batch_size: The global batch size for training (number of prompts per iteration).
            micro_batch_size: Micro batch size per device for training.
            n_samples_per_prompt: The number of samples generated per prompt.
            tokenizer: Tokenizer to use for text processing.
            dataset_additional_keys: Additional keys to include in the dataset.
            blocking: Whether to enable blocking mode for remote calls.
            guarantee_order: Whether to guarantee generation order.
            num_cpus_for_local_task: Number of CPUs for local ray task.
            partial_rollout_max_split: Maximum split for partial rollout generation.
            **kwargs: Additional parameters for base class argument passing.
        """
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
            partial_rollout_max_split=partial_rollout_max_split,
            **kwargs
        )

        # Transfer dock for experience data communication
        self.transfer_dock = None
        # Transfer dock for sampling data communication
        self.sampling_transfer_dock = None
        # Transfer dock for multi-modal data communication
        self.mm_transfer_dock = None
        # Transfer dock for multi-modal sampling data communication
        self.mm_sampling_transfer_dock = None
        # Metrics collector for training statistics
        self.metrics = Metric()
        # Data strategy manager for batch processing and filtering
        self.data_strategy = DataStrategy(self.actor_worker.rl_config)
        # Additional keyword arguments
        self.kwargs = kwargs
        # Whether to enable dynamic sampling filtering
        self.should_filter = kwargs['filter_groups_enable']
        # Maximum number of prompts in a filtered batch
        self.max_num_prompt_in_batch = kwargs['filter_groups_train_batch_size']
        # Maximum number of generation batches for filtering
        self.max_num_gen_batches = kwargs['filter_groups_max_batches']
        # Current number of prompts in batch
        self.num_prompt_in_batch = 0
        # Current number of generation batches
        self.num_gen_batches = 0
        # List of dynamic sampling controllers
        self.dynamic_sampling_list = dynamic_sampling_list
        # Additional columns to include in experience data
        self.addition_columns = ['metric_for_dapo']
        # Consumers for additional columns
        self.addition_consumers = ["dynamic_sampling", "dapo_metrics"]
        if self.dataset_additional_keys:
            self.addition_columns.extend(self.dataset_additional_keys)
        # Whether multi-turn conversation is enabled
        self.multi_turn_enable = kwargs['multi_turn_enable']
        if self.multi_turn_enable:
            self.addition_columns.extend(['response_mask', 'tool_call_num'])
        # Whether partial rollout is enabled
        self.enable_partial_rollout = self.partial_rollout_max_split > 1
        # Maximum length of transfer dock queue
        if self.enable_partial_rollout:
            self.td_max_len = self.global_batch_size * 2
        else:
            self.td_max_len = self.global_batch_size

        self.transfer_dock_init()
        self.set_actor_log_prob_skip_flag()

    def transfer_dock_init(self):
        """
        Initialize transfer docks for data communication between workers.
        
        Builds the DAPO data strategy and synchronizes transfer dock references
        with actor worker, reference worker, and reward workers.
        """
        self.data_strategy.build_dapo(
            should_filter=self.should_filter,
            max_num_prompt_in_batch=self.max_num_prompt_in_batch,
            td_max_len=self.td_max_len,
            n_samples_per_prompt=self.n_samples_per_prompt,
            metrics=self.metrics,
            max_age=self.partial_rollout_max_split,
            gbs_train=self.global_batch_size,
            addition_columns=self.addition_columns,
            addition_consumers=self.addition_consumers,
            dataset_additional_keys=self.dataset_additional_keys,
        )
        self.transfer_dock = self.data_strategy.td
        self.sampling_transfer_dock = self.data_strategy.sampling_td
        self.mm_transfer_dock = self.data_strategy.mm_td
        self.mm_sampling_transfer_dock = self.data_strategy.mm_sampling_td
        if self.should_filter:
            for sampling in self.dynamic_sampling_list:
                sampling.init_transfer_dock.remote(
                    self.transfer_dock,
                    self.mm_transfer_dock,
                    self.sampling_transfer_dock,
                    self.mm_sampling_transfer_dock,
                )

        self.actor_worker.sync_init_transfer_dock(self.transfer_dock, self.mm_transfer_dock, self.sampling_transfer_dock, self.mm_sampling_transfer_dock)
        if self.ref_worker:
            self.ref_worker.sync_init_transfer_dock(self.transfer_dock, self.mm_transfer_dock, self.sampling_transfer_dock, self.mm_sampling_transfer_dock)
        for reward in self.reward_list:
            if hasattr(reward, 'sync_init_transfer_dock'):
                reward.sync_init_transfer_dock(self.transfer_dock, self.mm_transfer_dock, self.sampling_transfer_dock, self.mm_sampling_transfer_dock)
            else:
                reward.init_transfer_dock.remote(self.transfer_dock, self.mm_transfer_dock, self.sampling_transfer_dock, self.mm_sampling_transfer_dock)

    def set_actor_log_prob_skip_flag(self):
        """
        Determine whether to skip actor log probability computation.
        
        Sets the skip flag when batch size matches mini-batch size and only one epoch,
        avoiding redundant computation in specific configurations.
        """
        if self.should_filter:
            global_batch_size = self.max_num_prompt_in_batch
        else:
            global_batch_size = self.global_batch_size
        mini_batch_size = self.actor_worker.rl_config.mini_batch_size
        epochs = self.actor_worker.rl_config.epochs
        self.skip_actor_log_prob = (global_batch_size * self.n_samples_per_prompt == mini_batch_size and epochs == 1)
        self.actor_worker.skip_actor_log_prob = self.skip_actor_log_prob

    def put_experience_data(self, batch, data_num, add_another_batch):
        """
        Put experience data into transfer dock for training.
        
        Args:
            batch: Raw data batch from data loader.
            data_num: Number of data samples to process.
            add_another_batch: Whether to add another batch to existing data.
        """
        if self.should_filter:
            self.data_strategy.clear_sampling(consumer='dynamic_sampling')
            index_list = self.data_strategy.prefetch_sampling_index(data_num)
            if index_list:
                if is_multimodal():
                    self.data_strategy.clear_mm(sampling=True)
                    self.data_strategy.put_mm(batch, indexes=[i for i in range(len(batch['prompts']) * self.n_samples_per_prompt)],
                                              sampling=True)
                batch, indexes = put_prompts_experience(batch, self.n_samples_per_prompt, self.dataset_additional_keys,
                                                        indexes=index_list, add_another_batch=add_another_batch)
                self.data_strategy.put_sampling(batch, indexes, is_prompt=True)
        else:
            if is_multimodal():
                self.data_strategy.clear_mm()
                self.data_strategy.put_mm(batch, indexes=[i for i in range(len(batch['prompts']) * self.n_samples_per_prompt)])
            batch, indexes = put_prompts_experience(batch, self.n_samples_per_prompt, self.dataset_additional_keys,
                                                    add_another_batch=add_another_batch)
            self.data_strategy.put_experience(batch, indexes, is_prompt=True)

    def fit(self, data_loader):
        """
        Main training loop for DAPO.
        
        Args:
            data_loader: PyTorch DataLoader providing training data batches.
        """
        logger = Loggers('dapo_trainer_hybrid')
        metrics = Metric()

        data_iters = iter(data_loader)
        data_iters_max_num = len(data_loader)
        data_iters_cur_num = 0
        iteration = self.actor_worker.get_iteration()

        if self.blocking:
            logger.info('sync start dapo training at iteration: {}/{} ...'.format(iteration, self.train_iters))
        else:
            logger.info('async start dapo training at iteration: {}/{} ...'.format(iteration, self.train_iters))

        data_num = self.global_batch_size * self.n_samples_per_prompt
        if self.enable_partial_rollout:
            first_batch = next(data_iters)
            data_iters_cur_num += 1
            self.put_experience_data(first_batch, data_num, add_another_batch=False)

        all_time = 0
        while iteration < self.train_iters:
            start_time = time.time()
            if data_iters_cur_num == data_iters_max_num:
                logger.info(f"dapo fit refresh data_iters")
                data_iters = iter(data_loader)
                data_iters_cur_num = 0

            batch = next(data_iters)
            data_iters_cur_num += 1

            last_iter = iteration == self.train_iters - 1
            data_num = self.global_batch_size * self.n_samples_per_prompt
            if self.enable_partial_rollout:
                if not last_iter:
                    self.put_experience_data(batch, data_num, add_another_batch=True)
            else:
                self.put_experience_data(batch, data_num, add_another_batch=False)
            
            # generate sequences
            logger.info(f"dapo fit generate_sequences")
            if self.should_filter:
                self.actor_worker.enter_infer_mode(blocking=self.blocking)

            if self.enable_partial_rollout and not self.skip_actor_log_prob:
                self.actor_worker.generate_sequences(blocking=True)
            else:
                self.actor_worker.generate_sequences(blocking=self.blocking)

            # compute rm scores.
            logger.info(f"dapo fit compute_rm_score")
            rule_reward = []
            for reward_worker in self.reward_list:
                if isinstance(reward_worker, RayActorGroup):
                    reward_worker.compute_rm_score(blocking=self.blocking)
                else:
                    rule_reward.append(reward_worker.compute_rm_score.remote())

            if self.should_filter:
                # dynamic sampling
                logger.info(f"dapo fit dynamic_sampling")
                should_continue = self.dynamic_sampling()
                if should_continue:
                    end_time = time.time()
                    all_time += end_time - start_time
                    continue

                self.actor_worker.exit_infer_mode(blocking=self.blocking)
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
                                                          self.guarantee_order,
                                                          self.multi_turn_enable)
            metrics_result = self.data_strategy.get_metrics()
            end_time = time.time()
            all_time += end_time - start_time

            metrics = self.process_metric(all_time, metrics_result, dapo_data_metrics, metrics)
            iteration += 1
            all_time = 0
            logger.info(metrics.metric, iteration, self.train_iters)
            if self.tensorboard is not None:
                for k, v in metrics.metric.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iteration)
            if self.wandb is not None:
                self.wandb.log_metrics(metrics.metric, iteration)
            if self.swanlab is not None:
                self.swanlab.log_metrics(metrics.metric, iteration)

            if iteration % self.save_interval == 0 or iteration == self.train_iters:
                self.save_checkpoint(iteration)

            self.data_strategy.clear_main()
            if self.should_filter:
                logger.info(f"dapo fit clear")
                # Refresh current train prompt num, compare with filter_groups_train_batch_size
                self.num_prompt_in_batch = 0
                # Refresh current gen batch num, compare with filter_groups_max_batches
                self.num_gen_batches = 0

        logger.info('after dapo training is done')
        ray.shutdown()

    def dynamic_sampling(self):
        """
        Perform dynamic sampling to filter high-quality training data.
        
        Increments generation batch counter, executes remote sampling workers,
        and validates if collected data meets filtering thresholds.
        
        Returns:
            bool: True if sampling should continue (insufficient data), 
                  False if enough data is collected.
                  
        Raises:
            ValueError: If maximum generation batches exceeded without collecting enough data.
        """
        logger = Loggers("dynamic_sampling")
        self.num_gen_batches += 1

        sampling_list = []
        for sampling in self.dynamic_sampling_list:
            sampling_list.append(sampling.dynamic_sampling.remote())
        ray.get(sampling_list)
        experience_data_num = self.data_strategy.get_cur_index()
        self.num_prompt_in_batch = experience_data_num // self.n_samples_per_prompt
        logger.info(f"dynamic_sampling: num_prompt_in_batch {self.num_prompt_in_batch}")

        if self.num_prompt_in_batch < self.max_num_prompt_in_batch:
            if self.max_num_gen_batches <= 0 or self.num_gen_batches < self.max_num_gen_batches:
                return True
            else:
                raise ValueError('Generated too many. Please check your data.')

        return False

    def compute_advantage(self, data_num, blocking=False, guarantee_order=False):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            data_num: Number of data samples to compute advantages for.
            blocking: Whether to block until computation completes.
            guarantee_order: Whether to guarantee data order in computation.
        """
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
            n_sample_per_prompt=self.n_samples_per_prompt,
            multi_turn_enable=self.multi_turn_enable
        )
        if blocking:
            ray.get(compute_advantage_ref)
        end_adv_time = time.time()
        self.data_strategy.update_metrics(
            "timing/adv",
            value=[round(end_adv_time, 4), round(start_adv_time, 4)],
            cumulate=True
        )
        self.data_strategy.update_metrics(
            "end_time/end_adv_time",
            value=[round(end_adv_time, 4)],
            cumulate=True
        )

    def process_metric(self, all_time, metrics_result, dapo_data_metrics, metrics):
        """
        Process and aggregate training metrics.
        
        Args:
            all_time: Total elapsed time for the iteration.
            metrics_result: Raw metrics from data strategy.
            dapo_data_metrics: DAPO-specific data metrics.
            metrics: Metrics collector object to update.
            
        Returns:
            Metric: Updated metrics collector with processed statistics.
        """
        if self.should_filter:
            enter_infer_time_list = metrics_result.metric.get("timing/resharding_enter_infer", [0])
            exit_infer_time_list = metrics_result.metric.get("timing/resharding_exit_infer", [0])
            resharding_to_infer_time_list = []
            for enter_infer_time, exit_infer_time in zip(enter_infer_time_list, exit_infer_time_list):
                resharding_to_infer_time = enter_infer_time + exit_infer_time
                resharding_to_infer_time_list.append(resharding_to_infer_time)
            metrics_result.update("timing/resharding_to_infer", resharding_to_infer_time_list)
        metrics_result = metrics_post_processing(metrics_result)
        metrics_result = metrics_sort(metrics_result, all_time)
        metrics.update(value=metrics_result)
        metrics.update(value=dapo_data_metrics)

        if self.should_filter:
            metrics.update("train/num_gen_batches", self.num_gen_batches)
            tps = compute_tps(self.kwargs, dapo_data_metrics, self.global_batch_size * self.num_gen_batches,
                              self.n_samples_per_prompt, all_time)
            update_tps = compute_tps(self.kwargs, dapo_data_metrics, self.max_num_prompt_in_batch,
                                     self.n_samples_per_prompt, metrics_result["timing/update"])
            vllm_tps = compute_tps(self.kwargs, dapo_data_metrics, self.global_batch_size * self.num_gen_batches,
                                   self.n_samples_per_prompt, metrics.metric["timing/rollout"])
        else:
            tps = compute_tps(self.kwargs, dapo_data_metrics, self.global_batch_size,
                              self.n_samples_per_prompt, all_time)
            update_tps = compute_tps(self.kwargs, dapo_data_metrics, self.global_batch_size,
                                     self.n_samples_per_prompt, metrics_result["timing/update"])
            vllm_tps = compute_tps(self.kwargs, dapo_data_metrics, self.global_batch_size,
                                   self.n_samples_per_prompt, metrics_result["timing/rollout"])
        metrics.update("e2e_tps", tps)
        metrics.update("update_tps", update_tps)
        metrics.update("vllm_tps", vllm_tps)

        metrics.remove_key("timing/non_overlap_reference_model")
        metrics.remove_key("timing/non_overlap_rule_reward")
        metrics.remove_key("timing/rule_reward")
        metrics.remove_key("actor/kl_loss")

        return metrics

    def save_checkpoint(self, iteration: int):
        self.actor_worker.save_checkpoint(iteration)