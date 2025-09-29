# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import ray
from transformers import AutoTokenizer
import torch

from mindspeed_rl.models.rule_verifier import compute_verifier_score, math_compute_score, math_acc_reward
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.trainer.utils.transfer_dock import pad_experience
from mindspeed_rl.utils.pad_process import remove_padding_tensor_dict_to_dict, padding_dict_to_tensor_dict
from mindspeed_rl.utils.utils import get_current_dp_range_indexes, is_multimodal, get_node_nums

logger = Loggers("rule_reward")


@ray.remote
class RuleReward(object):

    def initialize(self, megatron_config, rl_config, tokenizer, trust_remote_code=False, dp_rank=0):
        self.rl_config = rl_config
        self.megatron_config = megatron_config
        self.n_samples_per_prompt = rl_config.n_samples_per_prompt
        self.tokenizer = tokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(megatron_config.tokenizer_name_or_path,
                                                          trust_remote_code=trust_remote_code)
        self.dp_rank = dp_rank

    def init_transfer_dock(self, td, mm_td=None, sampling_transfer_dock=None, mm_sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock
        self.mm_sampling_transfer_dock = mm_sampling_transfer_dock

    def compute_rm_score(self):
        experience_consumer_stage = 'rule_reward'
        experience_columns = ['prompts', 'responses', 'response_length', *self.megatron_config.dataset_additional_keys]
        experience_count = self.rl_config.reward_dispatch_size
        assign_batch_size = self.megatron_config.global_batch_size * self.rl_config.n_samples_per_prompt // get_node_nums()
        sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                      assign_batch_size=assign_batch_size, current_dp_rank=self.dp_rank) if self.rl_config.guarantee_order else None

        pad_token_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
        cur_td = self.sampling_transfer_dock if self.sampling_transfer_dock else self.td

        while not ray.get(cur_td.all_consumed.remote(experience_consumer_stage)):
            batch_data, index = ray.get(
                cur_td.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                    get_n_samples=True
                )
            )  # cpu数据
            batch_data = remove_padding_tensor_dict_to_dict(batch_data)
            if batch_data and index:
                batch_data = pad_experience(batch_data, pad_token_id) # multiple, tp_size
                if not is_multimodal():
                    if "categories" in batch_data.keys():
                        use_verifier_mask = batch_data["categories"][:, 0].squeeze().bool()
                        selected_index = [index[i] for i in range(len(index)) if use_verifier_mask[i]]
                        index = selected_index
                    if not index:
                        continue
                    if "categories" in batch_data.keys():
                        batch_data = {key: value[use_verifier_mask] if key != 'prompts' else value[
                            use_verifier_mask[::self.n_samples_per_prompt]] for key, value in batch_data.items()}
                    ignore_token = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod

                    rm_scores, metrics = compute_verifier_score(
                        batch_data,
                        self.megatron_config,
                        self.rl_config,
                        self.hf_tokenizer,
                        ignore_token
                    )

                    for key, value in metrics.items():
                        ray.get(self.td.update_metrics.remote(key, value=value, cumulate=True))

                    output = {"rm_scores": rm_scores}
                    if self.rl_config.filter_groups_enable:
                        metric = torch.tensor(metrics[self.rl_config.filter_groups_metric], dtype=torch.float32,
                                            device=rm_scores.device)
                        metric = metric.reshape(rm_scores.shape)
                        output["metric_for_dapo"] = metric
                    logger.info("finish compute scores")
                    output = padding_dict_to_tensor_dict(output)
                    cur_td.put_experience.remote(data_dict=output, indexes=index)
                else:
                    mm_cur_td = self.mm_sampling_transfer_dock if self.mm_sampling_transfer_dock else self.mm_td
                    mm_columns = ray.get(mm_cur_td.get_columns.remote(experience_consumer_stage))
                    batch_mm_data = ray.get(mm_cur_td.get_experience.remote(mm_columns, index))
                    batch_data.update(batch_mm_data)

                    reward_tensor = torch.zeros((batch_data['responses'].size(0), 1), dtype=torch.float32)
                    original_shape = reward_tensor.shape
                    responses = batch_data['responses']
                    response_strs = self.hf_tokenizer.batch_decode(responses, skip_special_tokens=True)
                    labels = []
                    for _ in range(self.n_samples_per_prompt):
                        for label in batch_data['labels']:
                            labels.append(label)

                    metrics_score = []
                    for i, (response_str, label) in enumerate(zip(response_strs, labels)):
                        token_level_rewards = math_compute_score(response_str, label)
                        reward_tensor[i, 0] = token_level_rewards
                        metrics_score.append(int(math_acc_reward(response_str, label)))
                    metrics = {"acc_for_dapo_rewards/mean": metrics_score}
                    rm_scores = reward_tensor
                    reward_tensor_reshaped = reward_tensor.reshape(-1, self.n_samples_per_prompt)
                    reward_mean = reward_tensor_reshaped.mean(dim=1, keepdim=True)
                    reward_std = reward_tensor_reshaped.std(dim=1, keepdim=True) + 1e-6
                    reward_tensor_normalized = (reward_tensor_reshaped - reward_mean) / reward_std
                    reward_tensor = reward_tensor_normalized.reshape(original_shape)
                    output = {"rm_scores": rm_scores, "token_level_rewards": reward_tensor}
                    if self.rl_config.filter_groups_enable:
                        metric = torch.tensor(metrics[self.rl_config.filter_groups_metric], dtype=torch.float32,
                                            device=rm_scores.device)
                        metric = metric.reshape(rm_scores.shape)
                        output["metric_for_dapo"] = metric
                    output = padding_dict_to_tensor_dict(output)
                    cur_td.put_experience.remote(data_dict=output, indexes=index)
