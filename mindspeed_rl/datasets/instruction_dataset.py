# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import warnings
from typing import Optional, Callable, List, Any

import torch
import numpy as np

from mindspeed_rl.datasets.templates import get_model_template
from mindspeed_rl.datasets.utils import _infer_seqlen, get_prompt_index, _build_index_mappings
from mindspeed_rl.datasets.base_dataset import BaseDataset
from mindspeed_rl.datasets.indexed_dataset import get_packed_indexed_dataset

warnings.filterwarnings("ignore")


class InstructionDataset(BaseDataset):
    """InstructionDataset.

    Args:
        parallel_state: Megatron parallel state
        dataset_type: data type (default is LLM)
        data_prefix: path+prefix for data
        is_packed_data: True supported only
        tokenizer: tokenizer by get_tokenizer
        seq_length: teh length of sequence (default is 2048)
        num_samples: minimal samples
        name: name for index file
        documents: a list with the range that gives the number of documents
        seed: random seed
        full_shuffle_instruction_dataset: full shuffle for all index
        no_shuffle: do not use shuffle index
        reset_position_ids: support for TND Training
        prompt_type: for instruction training, model related
        prompt_type_path: the path to templates.json
    """
    def __init__(self,
                 parallel_state: Any,
                 dataset_type: str = "LLM",
                 data_prefix: str = "",
                 is_packed_data: bool = True,
                 tokenizer: Callable = None,
                 seq_length: int = 2048,
                 num_samples: int = None,
                 name: str = "",
                 documents: List[int] = None,
                 seed: int = 42,
                 full_shuffle_instruction_dataset: bool = False,
                 no_shuffle: bool = False,
                 reset_position_ids: bool = False,
                 prompt_type: str = None,
                 prompt_type_path: str = None,
                 extra_param: Optional[Any] = None
                 ):
        self.parallel_state = parallel_state
        self.data_prefix = data_prefix
        self.is_packed_data = is_packed_data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.name = name
        self.documents = documents
        self.seed = seed
        self.full_shuffle_instruction_dataset = full_shuffle_instruction_dataset
        self.no_shuffle = no_shuffle
        self.reset_position_ids = reset_position_ids
        self.prompt_type = prompt_type
        self.prompt_type_path = prompt_type_path

        if self.is_packed_data:
            dataset = get_packed_indexed_dataset(data_prefix=self.data_prefix)

            self.shuffle_index = _build_index_mappings(name=self.name,
                                                       data_prefix=self.data_prefix,
                                                       start_index=self.documents[0],
                                                       nb_documents=len(self.documents),
                                                       num_samples=self.num_samples,
                                                       seed=self.seed,
                                                       full_shuffle_instruction_dataset=full_shuffle_instruction_dataset,
                                                       parallel_state=self.parallel_state)

            self.dataset_type = "Preference_DS_Packed"
        else:
            raise NotImplementedError('non packed data are not supported yet.')

        super().__init__(dataset, dataset_type, extra_param)

    def __len__(self):
        return len(self.shuffle_index)

    def __getitem__(self, idx):
        if self.is_packed_data:
            doc_idx = self.shuffle_index[idx]

            if self.no_shuffle:
                doc_idx = idx % len(self.dataset)

            item = self.dataset[doc_idx]

            # pack模式下固定长度为seq_length，不需要cut
            if self.reset_position_ids:
                position_ids = self._get_reset_position_ids(torch.from_numpy(item['input_ids']))
                return {
                    "input_ids": self._cut_token(item['input_ids'], np.int64),
                    "attention_mask": self._cut_token(item["attention_mask"], np.int64),
                    "labels": self._cut_token(item["labels"], np.int64),
                    "position_ids": self._cut_token(position_ids.numpy(), np.int64)
                }
            else:
                return self._cut_instruction_token(item, np.int64)
        else:
            raise NotImplementedError('non packed data are not supported yet.')

    def _cut_instruction_token(self, item, dtype):
        IGNORE_INDEX = -100
        if "labels" in item.keys():
            token_length = len(item["input_ids"])
            if token_length <= self.seq_length:
                return {
                    "input_ids": item["input_ids"].astype(dtype),
                    "attention_mask": np.ones_like(item["input_ids"]).astype(dtype),
                    "labels": item["labels"].astype(dtype)
                }

            template = None
            # get model chat template
            if self.prompt_type is not None and self.prompt_type_path:
                template = get_model_template(self.prompt_type, self.prompt_type_path)

            prompt_begin_list, prompt_end_list = get_prompt_index(item["labels"], IGNORE_INDEX)

            multi_turns = len(prompt_begin_list)
            total_length = 0

            if template is not None and template.efficient_eos:
                total_length = 1
                prompt_end_list = [x - 1 for x in prompt_end_list]
                eos_token_id = item["input_ids"][token_length - 1]
                item["input_ids"] = item["input_ids"][:token_length]
                item["labels"] = item["labels"][:token_length]

            cutoff_len = self.seq_length
            input_ids = np.array([], dtype=dtype)
            labels = np.array([], dtype=dtype)

            for turn_idx in range(multi_turns):
                if total_length >= cutoff_len:
                    break
                source_ids = item["input_ids"][prompt_begin_list[turn_idx]:prompt_end_list[turn_idx]]
                mask_ids = item["labels"][prompt_begin_list[turn_idx]:prompt_end_list[turn_idx]]

                label_begin_idx = prompt_end_list[turn_idx]

                if turn_idx != multi_turns - 1:
                    target_ids = item["labels"][label_begin_idx:prompt_begin_list[turn_idx + 1]]
                else:
                    target_ids = item["labels"][label_begin_idx:]

                source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)

                source_ids = source_ids[:source_len]
                target_ids = target_ids[:target_len]
                mask_ids = mask_ids[:source_len]

                total_length += source_len + target_len
                input_ids = np.concatenate((input_ids, source_ids, target_ids), axis=0)
                labels = np.concatenate((labels, mask_ids, target_ids), axis=0)

            if template is not None and template.efficient_eos:
                input_ids = np.concatenate((input_ids, np.array([eos_token_id], dtype=dtype)), axis=0)
                labels = np.concatenate((labels, np.array([eos_token_id], dtype=dtype)), axis=0)

            res = {
                "input_ids": input_ids.astype(dtype),
                "attention_mask": np.ones_like(input_ids).astype(dtype),
                "labels": labels.astype(dtype)
            }

        else:
            prompt_ids = item["input_ids"]
            input_ids = prompt_ids[:self.seq_length]

            res = {
                "input_ids": input_ids.astype(dtype),
                "attention_mask": np.ones_like(input_ids).astype(dtype)
            }

        return res

    def _get_reset_position_ids(self, data: torch.Tensor):
        seq_length = data.numel()
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)

        # Find indices where EOD token is.
        eod_index = position_ids[data == self.tokenizer.eod]
        # Detach indices from positions if going to modify positions.

        eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Reset positions.
            position_ids[(i + 1):] -= i + 1 - prev_index
            prev_index = i + 1

        return position_ids.clone()

    def _cut_token(self, token, dtype):
        token_length = len(token)
        if token_length >= self.seq_length:
            token = token[:self.seq_length]
        return token.astype(dtype)
