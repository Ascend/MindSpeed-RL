# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import numpy as np

from mindspeed_rl import get_tokenizer
from mindspeed_rl import InstructionDataset, InstructionDataLoader

from tests.test_tools.dist_test import DistributedTest


class TestInstructionDataset(DistributedTest):
    world_size = 1

    def test_nonpack_dataset(self):
        tokenizer_directory = '/data/models/llama2-7b'
        non_pack_data_prefix = '/data/datasets/nonpack/alpaca'

        hf_tokenizer = get_tokenizer(tokenizer_directory)

        from megatron.core import parallel_state
        documents = np.arange(start=0, stop=52002, step=1, dtype=np.int32)

        dataset = InstructionDataset(
            parallel_state=parallel_state,
            dataset_type='LLM',
            data_prefix=non_pack_data_prefix,
            is_packed_data=True,
            tokenizer=hf_tokenizer,
            seq_length=1024,
            num_samples=1000,
            name="test",
            documents=documents,
            seed=42,
            extra_param=None,
            full_shuffle_instruction_dataset=False,
            no_shuffle=False,
            reset_position_ids=False,
            prompt_type='llama2',
            prompt_type_path='./configs/templates.json'
        )

        dataloader = InstructionDataLoader(
            dataset=dataset,
            parallel_state=parallel_state,
            tokenizer=None,
            num_workers=2,
            tokenizer_padding_side='right',
            pad_to_multiple_of=8,
            variable_seq_lengths=False,
            num_nextn_predict_layers=0,
            micro_batch_size=1,
            comsumed_samples=0,
            seed=1234
        )

        for item in dataloader:
            assert item['input_ids'][0][0] == 128000, "non packed data failed!"
            assert item['labels'][0][-1] == -100, "non packed data failed!"
            break

    def test_pack_dataset(self):
        tokenizer_directory = '/data/models/llama2-7b'
        pack_data_prefix = '/data/datasets/pack/alpaca'

        hf_tokenizer = get_tokenizer(tokenizer_directory)

        from megatron.core import parallel_state
        documents = np.arange(start=0, stop=1038, step=1, dtype=np.int32)

        dataset = InstructionDataset(
            parallel_state=parallel_state,
            dataset_type='LLM',
            data_prefix=pack_data_prefix,
            is_packed_data=True,
            tokenizer=hf_tokenizer,
            seq_length=1024,
            num_samples=1000,
            name="test",
            documents=documents,
            seed=42,
            extra_param=None,
            full_shuffle_instruction_dataset=False,
            no_shuffle=False,
            reset_position_ids=True,
            prompt_type='llama2',
            prompt_type_path='./configs/templates.json',
        )

        def dummy_dp_size():
            return 1

        parallel_state.get_data_parallel_world_size = dummy_dp_size
        dataloader = InstructionDataLoader(
            dataset=dataset,
            parallel_state=parallel_state,
            tokenizer=None,
            num_workers=2,
            tokenizer_padding_side='right',
            pad_to_multiple_of=8,
            variable_seq_lengths=False,
            num_nextn_predict_layers=0,
            micro_batch_size=1,
            comsumed_samples=0,
            seed=1234
        )

        for item in dataloader:
            assert item['input_ids'][0][-1] == 387, "non packed data failed!"
            assert item['labels'][0][0] == -100, "non packed data failed!"
            break
