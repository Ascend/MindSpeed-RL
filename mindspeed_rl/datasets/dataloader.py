# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Sequence, Dict, Any

import torch

from .data_samplers import PromptSampler


class PromptDataLoader(torch.utils.data.DataLoader):
    """PromptDataLoader.

    Args:
        dataset: An Prompt Implementation of BaseDataset
        consumed_samples: the number of consumed samples for continue training
        global_batch_size: global batch size for loader
        num_workers: workers of dataloader
        seed: random seed
        dataset_additional_keys: extra keys for data loading
    """
    def __init__(self,
                 dataset,
                 consumed_samples,
                 global_batch_size,
                 num_workers,
                 seed,
                 dataset_additional_keys):


        batch_sampler = PromptSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            batch_size=global_batch_size)

        def collator(features, return_tensors=None):
            features_dict = {}

            features_dict["prompts"] = [torch.tensor(value['input_ids']) for value in features]

            for add_key in dataset_additional_keys:
                features_dict[add_key] = [torch.tensor(value[add_key]) for value in features]

            return features_dict

        super().__init__(dataset,
                        batch_sampler=batch_sampler,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=collator,
                        pin_memory=True
                         )
