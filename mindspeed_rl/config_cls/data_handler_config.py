# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import os
from pathlib import Path
from typing import Optional

from mindspeed_rl.config_cls.base_config import BaseConfig

cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir, "../../configs/templates.json")


class DataHandlerConfig(BaseConfig):
    def __init__(self, config_dict):
        # input data parameters
        # Path to input JSON or path or a huggingface dataset name; for merge datasets, it is the directory path containing all document files to merge (required)
        self.input: str = None

        # specify a dataset handler
        self.handler_name: str = ""

        # weather to use streaming
        self.streaming: bool = False

        # space separate listed of keys to extract from json
        self.json_keys: list = ['text']

        # Split documents into sentences.
        self.split_sentences: bool = False

        # Keep newlines between sentences when splitting.
        self.keep_newlines: bool = False

        # Which template to use for constructing prompts in training. e.g., "qwen"
        self.prompt_type: str = "empty"

        # Path to the json file of templates.
        self.prompt_type_path: str = TEMPLATES_DIR

        # Additional keys need to be added from dataset.
        self.dataset_additional_keys: list = []

        # Probabilities to sample data from datasets. Use commas to separate multiple datasets. probabilities should sum to 1. ex: "0.1, 0.2, 0.3, 0.4"
        self.interleave_probs: Optional[str] = None

        # Overwrite the cached training and evaluation sets.
        self.overwrite_cache: bool = False

        # Random seed to be used with data mix.
        self.seed: int = 1234

        # Directory to store the cached dataset locally.
        self.cache_dir: str = os.path.join(os.path.expanduser("~"), "cache")

        # Dataset field mapping.
        self.map_keys: Optional[dict] = None

        # Package multiple samples into one sample in a fine-tuning dataset
        self.pack: bool = False

        # Use a zigzag attention mask.
        self.neat_pack: bool = False

        # Python script dataset direction
        self.script_data_dir: Optional[str] = None

        # tokenizer parameters
        # What type of tokenizer to use.
        self.tokenizer_type: str = 'HuggingFaceTokenizer'

        # HuggingFace tokenizer not use the fast version.
        self.tokenizer_not_use_fast: bool = True

        # Path to the vocab file
        self.vocab_file: Optional[str] = None

        # Path to the BPE merge file
        self.merge_file: Optional[str] = None

        # Append an <eod> token to the end of a document.
        self.append_eod: bool = False

        # Name or path of the huggingface tokenizer.
        self.tokenizer_name_or_path: str = None

        # Maximum sequence length to process.
        self.seq_length: int = None

        # Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.
        self.make_vocab_size_divisible_by: int = 128

        # Pad the vocab size to be divisible by this value. Value of the size of the vocabulary of the tokenizer to reach.
        # This value must be greater than the initial size of the tokenizer. If this argument is used the value of
        # `make-vocab-size-divisible-by` will be ignored.
        self.pad_vocab_size_to: int = None

        # A special placeholder token marking the end of each step where the PRM can make predictions.
        self.placeholder_token: str = "ки"

        # The labels represent the correctness of each reasoning step in the entire reasoning process.
        self.reward_tokens: list = []

        # Path to binary output file without suffix (required)
        self.output_prefix: str = None

        # Dataset storage format, options: ['lazy', 'cached', 'mmap']
        self.dataset_impl: str = "mmap"

        # Number of worker processes to launch
        self.workers: int = 1

        # Number of subsets to cut for multiprocessing
        self.n_subs: int = 1

        # Interval between progress updates
        self.log_interval: int = 100

        # The `bin-idx` pair files with the same key in their filename will be merged.
        self.merge_group_keys: list = None

        if config_dict is not None:
            self.update(config_dict)

        if self.input is None:
            raise ValueError("input is required.")

        if self.output_prefix is None:
            raise ValueError("output_prefix is required.")
