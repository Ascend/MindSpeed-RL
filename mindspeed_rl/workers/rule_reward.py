# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import time
import ray
from transformers import AutoTokenizer

from mindspeed_rl.models.rule_verifier import compute_verifier_score
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers("rule_reward")


@ray.remote
class RuleReward(object):

    def initialize(self, megatron_config, rl_config):
        self.rl_config = rl_config
        self.megatron_config = megatron_config
        self.n_samples_per_prompt = rl_config.n_samples_per_prompt

    def init_transfer_dock(self, td):
        self.td = td

    def compute_rm_score(self):
        experience_consumer_stage = 'rule_reward'
        experience_columns = ['prompts', 'responses', 'response_length', *self.megatron_config.dataset_additional_keys]
        experience_count = self.rl_config.experience_count_rule_reward
        tokenizer = AutoTokenizer.from_pretrained(self.megatron_config.tokenizer_name_or_path, trust_remote_code=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            batch_data, index = ray.get(
                self.td.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    pad_id=pad_id
                )
            )  # cpu数据

            if batch_data and index:
                if "categories" in batch_data.keys():
                    use_verifier_mask = batch_data["categories"][:, 0].squeeze().bool()
                    selected_index = [index[i] for i in range(len(index)) if use_verifier_mask[i]]
                    index = selected_index
                if not index:
                    continue
                if "categories" in batch_data.keys():
                    batch_data = {key: value[use_verifier_mask] if key != 'prompts' else value[
                        use_verifier_mask[::self.n_samples_per_prompt]] for key, value in batch_data.items()}
                token_level_rewards, metrics = compute_verifier_score(batch_data, self.megatron_config, self.rl_config)
                
                for key, value in metrics.items():
                    ray.get(self.td.update_metrics.remote(key, value=value, cumulate=True))

                output = {"rm_scores": token_level_rewards, "token_level_rewards": token_level_rewards}
                self.td.put_experience.remote(data_dict=output, indexes=index, num_responses=self.n_samples_per_prompt)
