from typing import Dict, Callable, Optional, Any

import numpy as np

import torch
from torch.utils.data import DataLoader

from mindspeed_rl.datasets.utils import _infer_seqlen, get_prompt_index

from mindspeed_rl.datasets.indexed_dataset import get_packed_indexed_dataset
from mindspeed_rl.datasets.base_dataset import BaseDataset
from mindspeed_rl.datasets.templates import get_model_template
from mindspeed_rl.datasets.utils import _build_index_mappings
from mindspeed_rl.datasets.data_samplers import PromptSampler


class PromptDataset(BaseDataset):
    def __init__(
            self,
            data_prefix: str = "",
            is_packed_data: bool = False,
            tokenizer: Callable = None,
            seq_length: int = 128,
            num_samples: int = None,
            name: str = "",
            documents: Any = None,
            seed: int = 42,
            full_shuffle_instruction_dataset: bool = False,
            token_param: Optional[Dict] = None,
            preprocess_template: Optional[str] = None,
            pad_token: int = 0,
            eos_token: int = 1,
            extra_param: Any = None,
            **kwargs,
    ):
        self.data_prefix = data_prefix
        self.is_packed_data = is_packed_data
        self.tokenizer = tokenizer
        self.token_param = token_param
        self.seq_length = seq_length
        self.preprocess_template = preprocess_template
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.num_samples = num_samples
        self.args = extra_param

        if self.is_packed_data:
            self.res_dataset = get_packed_indexed_dataset(data_prefix=self.data_prefix)
            self.shuffle_index = _build_index_mappings(name=name,
                                                       data_prefix=self.data_prefix,
                                                       start_index=documents[0],
                                                       nb_documents=len(documents),
                                                       num_samples=self.num_samples,
                                                       seed=seed,
                                                       full_shuffle_instruction_dataset=full_shuffle_instruction_dataset,
                                                       parallel_state=None,
                                                       no_shuffle=True)
            dataset_type = "Prompt_DS_Packed"
        else:
            raise NotImplementedError('non packed data are not supported yet.')

        super().__init__(self.res_dataset, dataset_type)

    def __len__(self):
        return len(self.shuffle_index)

    def __getitem__(self, index):
        doc_idx = self.shuffle_index[index]

        item = self.res_dataset[doc_idx]
        return self._cut_instruction_token(item, np.int64)

    def _cut_instruction_token(self, item, dtype):
        IGNORE_INDEX = -100
        if "labels" in item.keys() and not self.args.dataset_additional_keys:
            token_length = len(item["input_ids"])
            if token_length <= self.seq_length:
                return {
                    "input_ids": item["input_ids"].astype(dtype),
                    "attention_mask": np.ones_like(item["input_ids"]).astype(dtype),
                    "labels": item["labels"].astype(dtype)
                }

            template = None
            # get model chat template
            if hasattr(self.args, "prompt_type") and self.args.prompt_type is not None:
                template = get_model_template(self.args.prompt_type, self.args.prompt_type_path)

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

            add_vals = {}
            for add_keys in self.args.dataset_additional_keys:
                if add_keys in item.keys():
                    add_vals[add_keys] = item[add_keys]

            res = dict(
                {
                    "input_ids": input_ids.astype(dtype),
                    "attention_mask": np.ones_like(input_ids).astype(dtype)
                }, **add_vals
            )

        return res
