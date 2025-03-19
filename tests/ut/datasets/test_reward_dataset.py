# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import numpy as np

from megatron.core import parallel_state
from mindspeed_rl.datasets.reward_dataset import PreferenceDataset
from mindspeed_rl.datasets.dataloader import PreferenceDataLoader
from mindspeed_rl.utils.tokenizer import get_tokenizer

from tests.test_tools.dist_test import DistributedTest
from mindspeed_rl.config_cls.megatron_config import MegatronConfig


class TestRewardDataset(DistributedTest):
    world_size = 1

    def test_reward_dataset(self):
        tokenizer_directory = '/data/for_dt/tokenizer/Qwen25-7B'
        packed_data_prefix = '/data/for_dt/datasets/convert/orca_rlhf'
        hf_tokenizer = get_tokenizer(tokenizer_directory)
        dummy_config = MegatronConfig({}, {})
        dummy_config.stage = "orm"
        dummy_config.full_shuffle_instruction_dataset = False
        dummy_config.no_shuffle = True
        dummy_config.global_batch_size = 1
        documents = np.arange(start=0, stop=1024, dtype=np.int32)
        dataset = PreferenceDataset(
            parallel_state=parallel_state,
            data_prefix=packed_data_prefix,
            is_packed_data=True,
            tokenizer=hf_tokenizer,
            seq_length=1024,
            num_samples=12800,
            documents=documents
        )

        def dummy_dp_size():
            return 1

        parallel_state.get_data_parallel_world_size = dummy_dp_size
        dataloader = PreferenceDataLoader(dataset, parallel_state=parallel_state, num_workers=2, micro_batch_size=1)

        for item in dataloader:
            assert item['input_ids'][1][-1] == 151643
            break
