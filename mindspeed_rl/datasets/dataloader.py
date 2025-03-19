# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Sequence, Dict, Any

import torch
from transformers import DataCollatorForSeq2Seq

from .data_samplers import PretrainingSampler
from .data_samplers import PromptSampler


class InstructionDataLoader(torch.utils.data.DataLoader):
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


class PreferenceDataLoader(torch.utils.data.DataLoader):
    """PreferenceDataLoader.

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
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size()
        )

        if tokenizer is None:
            tokenizer = dataset.tokenizer

        tokenizer = tokenizer.tokenizer
        seq_length = dataset.seq_length
        tokenizer.tokenizer_padding_side = tokenizer_padding_side

        collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer,
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


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]], repeat=1) -> Dict[str, torch.Tensor]:
        """
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n * repeat (for hyper model) examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []

        for _ in range(repeat):
            self._concat(concatenated_features, features)

        return super().__call__(concatenated_features)

    @staticmethod
    def _concat(concatenated_features, features):
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }

                concatenated_features.append(target_feature)


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
