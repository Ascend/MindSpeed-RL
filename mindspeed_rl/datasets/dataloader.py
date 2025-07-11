# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Sequence, Dict, Any, List
from collections import defaultdict
from transformers import DataCollatorForSeq2Seq

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from .data_samplers import PretrainingSampler


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
                 global_batch_size,
                 num_workers,
                 seed,
                 dataset_additional_keys,
                 no_shuffle,
                 is_pairwise_dataset=False,
                 tokenizer=None):
        def collator(features, return_tensors=None):
            features_dict = {}

            features_dict["prompts"] = [torch.tensor(value['input_ids']) for value in features]

            for add_key in dataset_additional_keys:
                features_dict[add_key] = [torch.tensor(value[add_key]) for value in features]

            return features_dict

        if tokenizer is not None:
            collator_fn = PairwiseDataCollatorWithPadding(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors='pt',
                padding=True
            )

        if not no_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        super().__init__(dataset,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=collator_fn if is_pairwise_dataset else collator,
                        pin_memory=True,
                        sampler=sampler,
                        batch_size=global_batch_size,
                        drop_last=True)


class MultiModalDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 global_batch_size,
                 num_workers,
                 seed,
                 dataset_additional_keys,
                 no_shuffle):

        def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch_dict = defaultdict(list)
            for feature in features:
                for key, value in feature.items():
                    batch_dict[key].append(value)

            batch_dict['prompts'] = [torch.tensor(i) for i in batch_dict['prompts']]

            return batch_dict

        if not no_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        super().__init__(dataset,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=collate_fn,
                        pin_memory=True,
                        sampler=sampler,
                        batch_size=global_batch_size,
                        drop_last=True)


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]], repeat=2) -> Dict[str, torch.Tensor]:
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