# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import torch
from transformers import DataCollatorForSeq2Seq

from .data_samplers import PretrainingSampler
from .data_samplers import PromptSampler


class DataLoader(torch.utils.data.DataLoader):
    """DataLoader.

    Args:
        dataset: An Implementation of BaseDataset
        parallel_state: Megatron parallel state
        num_workers: workers of dataloader (default is 2)
        tokenizer: tokenizer by get_tokenizer
        tokenizer_padding_side: padding side for tokenizer
        pad_to_multiple_of: padding sequence when variable_seq_lengths is True (default is 8)
        variable_seq_lengths: variable seq length
        num_nextn_predict_layers: for MTP features
        micro_batch_size: micro batch size
        comsumed_samples: trained samples
        seed: random seed
    """
    def __init__(self,
                 dataset,
                 parallel_state,
                 num_workers=2,
                 tokenizer=None,
                 tokenizer_padding_side='right',
                 pad_to_multiple_of=8,
                 variable_seq_lengths=False,
                 num_nextn_predict_layers=0,
                 micro_batch_size=0,
                 comsumed_samples=0,
                 seed=1234):

        if dataset is None or len(dataset) == 0:
            raise ValueError('dataset is required and len(dataset) should be larger than 0.')

        batch_sampler = PretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=comsumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=0,
            data_parallel_size=1 if parallel_state.get_data_parallel_world_size()
                                    == 0 else parallel_state.get_data_parallel_world_size()
        )

        if tokenizer is None:
            tokenizer = dataset.tokenizer

        tokenizer = tokenizer.tokenizer
        seq_length = dataset.seq_length
        tokenizer.tokenizer_padding_side = tokenizer_padding_side

        collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of if variable_seq_lengths else seq_length + num_nextn_predict_layers,
            return_tensors='pt',
            padding=True
        )

        super().__init__(dataset,
                       batch_sampler=batch_sampler,
                       num_workers=num_workers,
                       generator=torch.Generator().manual_seed(seed),
                       collate_fn=collator,
                       pin_memory=True
                       )


class PromptDataLoader(torch.utils.data.DataLoader):
    def __init__(self, args, dataset, consumed_samples):
        batch_sampler = PromptSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            batch_size=args.global_batch_size)

        def collator(features, return_tensors=None):
            features_dict = {}

            features_dict["prompts"] = [torch.tensor(value['input_ids']) for value in features]

            for add_key in args.dataset_additional_keys:
                features_dict[add_key] = [torch.tensor(value[add_key]) for value in features]

            return features_dict

        super().__init__(dataset,
                        batch_sampler=batch_sampler,
                        num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=collator,
                        pin_memory=True
                         )
