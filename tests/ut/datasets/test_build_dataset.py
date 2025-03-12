# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from megatron.core import parallel_state

from mindspeed_rl import get_tokenizer, build_train_valid_test_datasets
from mindspeed_rl import InstructionDataset, DataLoader

from tests.test_tools.dist_test import DistributedTest


class TestBuildTrainValidTestDataset(DistributedTest):
    world_size = 1

    def test_build_nonpack_dataset(self):
        tokenizer_directory = '/data/models/llama2-7b'
        non_pack_data_prefix = '/data/datasets/nonpack/alpaca'

        hf_tokenizer = get_tokenizer(tokenizer_directory)

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=non_pack_data_prefix,
            splits_string='80,10,10',
            seq_length=1024,
            train_valid_test_num_samples=(3840, 5120, 1280),
            dataset_cls=InstructionDataset,
            tokenizer=hf_tokenizer,
            parallel_state=parallel_state,
            full_shuffle_instruction_dataset=False,
            no_shuffle=False,
            reset_position_ids=False,
            prompt_type='llama2',
            prompt_type_path='./configs/templates.json',
            seed=42,
            extra_param=None
        )

        train_dl = DataLoader(
            dataset=train_ds,
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

        for item in train_dl:
            assert item['input_ids'][0][-1] == 2, "build nonpack input_id failed!"
            assert item['labels'][0][-2] == -100, "build nonpack labels failed!"
            break

    def test_build_pack_dataset(self):
        tokenizer_directory = '/data/models/llama2-7b'
        pack_data_prefix = '/data/datasets/pack/alpaca'

        hf_tokenizer = get_tokenizer(tokenizer_directory)

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=pack_data_prefix,
            splits_string='100,0,0',
            seq_length=1024,
            train_valid_test_num_samples=(3840, 5120, 1280),
            dataset_cls=InstructionDataset,
            tokenizer=hf_tokenizer,
            parallel_state=parallel_state,
            full_shuffle_instruction_dataset=False,
            no_shuffle=False,
            reset_position_ids=True,
            prompt_type='llama2',
            prompt_type_path='./configs/templates.json',
            seed=42,
            extra_param=None
        )

        train_dl = DataLoader(
            dataset=train_ds,
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

        for item in train_dl:
            assert item['input_ids'][0][-3] == 1364, "build packed input_ids failed!"
            assert item['labels'][0][-1] == 387, "build packed labels failed!"
            break
