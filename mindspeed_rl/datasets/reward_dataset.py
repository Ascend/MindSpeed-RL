from typing import Any, Callable, List, Optional
import numpy as np

from mindspeed_rl.datasets.base_dataset import BaseDataset
from mindspeed_rl.datasets.utils import _infer_seqlen, _build_index_mappings
from mindspeed_rl.datasets.indexed_dataset import get_packed_indexed_dataset


class PreferenceDataset(BaseDataset):
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
            return self.cut_pairwise_token(item, np.int64)
        else:
            raise NotImplementedError('non packed data are not supported yet.')

    def cut_pairwise_token(self, item, dtype):
        # Cut prompt and response proportionally for pairwise datasets.
        IGNORE_INDEX = -100
        prompt_length = (item["chosen_labels"] != IGNORE_INDEX).nonzero()[0][0]
        prompt_ids = item["chosen_input_ids"][:prompt_length]
        chosen_ids = item["chosen_input_ids"][prompt_length:]
        rejected_ids = item["rejected_input_ids"][prompt_length:]
        source_len, target_len = _infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.seq_length
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = np.append(prompt_ids, chosen_ids)
        chosen_labels = np.append(IGNORE_INDEX * np.ones(source_len), chosen_ids)
        rejected_input_ids = np.append(prompt_ids, rejected_ids)
        rejected_labels = np.append(IGNORE_INDEX * np.ones(source_len), rejected_ids)

        res = {
            "chosen_input_ids": chosen_input_ids.astype(dtype),
            "chosen_attention_mask": np.ones_like(chosen_input_ids).astype(dtype),
            "chosen_labels": chosen_labels.astype(dtype),
            "rejected_input_ids": rejected_input_ids.astype(dtype),
            "rejected_attention_mask": np.ones_like(rejected_input_ids).astype(dtype),
            "rejected_labels": rejected_labels.astype(dtype)
        }

        return res
