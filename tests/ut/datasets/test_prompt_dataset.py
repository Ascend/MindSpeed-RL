# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import numpy as np
from torch.utils.data import SequentialSampler

from mindspeed_rl.datasets.prompt_dataset import PromptDataset
from mindspeed_rl.datasets.dataloader import PromptDataLoader
from mindspeed_rl.utils.tokenizer import get_tokenizer

from tests.test_tools.dist_test import DistributedTest
from mindspeed_rl.config_cls.megatron_config import MegatronConfig


class TestPromptDataset(DistributedTest):
    world_size = 1

    def test_prompt_dataset(self):
        tokenizer_directory = '/data/models/llama2-7b'
        packed_data_prefix = '/data/for_dt/datasets/pe-nlp/data'
        hf_tokenizer = get_tokenizer(tokenizer_directory)
        dummy_config = MegatronConfig({}, {})
        dummy_config.stage = "ray_ppo"
        dummy_config.full_shuffle_instruction_dataset = False
        dummy_config.global_batch_size = 1
        dummy_config.no_shuffle = True
        documents = np.arange(start=0, stop=3000, step=1, dtype=np.int32)
        dataset = PromptDataset(
            data_prefix=packed_data_prefix,
            is_packed_data=True,
            tokenizer=hf_tokenizer,
            seq_length=1024,
            num_samples=100,
            documents=documents,
            extra_param=dummy_config,
        )
        dataloader = PromptDataLoader(dataset, dummy_config.global_batch_size,
                                      dummy_config.num_workers, dummy_config.seed,
                                      dummy_config.dataset_additional_keys,
                                      dummy_config.no_shuffle)

        for item in dataloader:
            assert item['prompts'][0][0] == 151644
            break