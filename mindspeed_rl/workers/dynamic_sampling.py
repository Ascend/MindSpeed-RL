# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import ray
import numpy as np

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.utils import get_current_dp_range_indexes, extract_from_dict
from mindspeed_rl.utils.pad_process import remove_padding_tensor_dict_to_dict, padding_dict_to_tensor_dict

logger = Loggers("DynamicSampling")


@ray.remote
class DynamicSampling(object):
    def initialize(self, megatron_config, rl_config):
        self.rl_config = rl_config
        self.megatron_config = megatron_config
        self.n_samples_per_prompt = rl_config.n_samples_per_prompt
        self.global_batch_size = megatron_config.global_batch_size
        self.guarantee_order = rl_config.guarantee_order

    def init_transfer_dock(self, td, sampling_transfer_dock):
        self.td = td
        self.sampling_transfer_dock = sampling_transfer_dock

    def dynamic_sampling(self):
        experience_consumer_stage = 'dynamic_sampling'
        experience_columns = ['prompts', 'prompt_length', 'responses', 'response_length', 'input_ids', 'rm_scores',
                              'metric_for_dapo', *self.megatron_config.dataset_additional_keys]
        experience_count = self.rl_config.dynamic_sampling_dispatch_size
        assign_batch_size = self.global_batch_size * self.n_samples_per_prompt
        sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                      assign_batch_size=assign_batch_size) if self.guarantee_order else None
        while not ray.get(self.sampling_transfer_dock.all_consumed.remote(experience_consumer_stage)):
            batch_data, index = ray.get(
                self.sampling_transfer_dock.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    indexes=sorted_indexes.pop(0) if self.guarantee_order else None,
                    get_n_samples=True
                )
            )
            batch_data = remove_padding_tensor_dict_to_dict(batch_data)
            if batch_data and index:
                # filter by metric values
                metric_values = batch_data['metric_for_dapo']
                kept_idx_list = []
                for idx in range(0, experience_count, self.n_samples_per_prompt):
                    metric_group = metric_values[idx: idx + self.n_samples_per_prompt]
                    if np.std(metric_group) > 0 or len(metric_group) == 1:
                        kept_idx_list.extend(list(range(idx, idx + self.n_samples_per_prompt)))
                if not kept_idx_list:
                    logger.info(f"dynamic_sampling: kept_idx_list is empty")
                    continue

                index_list = ray.get(self.td.prefetch_request_index.remote(len(kept_idx_list)))
                if index_list:
                    kept_idx_list = kept_idx_list[:len(index_list)]
                    experience_data = extract_from_dict(batch_data, kept_idx_list)
                    experience_data = padding_dict_to_tensor_dict(experience_data)
                    ray.get(self.td.put_experience.remote(experience_data, index_list))

