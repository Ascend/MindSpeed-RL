# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
from pathlib import Path
from typing import Optional

from mindspeed_rl.config_cls.base_config import BaseConfig


cur_file_dir = Path(__file__).absolute().parent
TEMPLATES_DIR = os.path.join(cur_file_dir, "../../configs/model/templates.json")


class DataHandlerConfig(BaseConfig):
    """DataHandler configuration class for dataset processing and tokenization.

    This class manages data handling parameters including input/output paths,
    dataset preprocessing, tokenization configuration, and caching strategies
    for large-scale language model training.

    Attributes:
        input (str): Path to input data. Supported formats:
            - Path to JSON file
            - Path to directory containing dataset files
            - HuggingFace dataset name
            - Directory path for merge datasets (containing all document files to merge).
        handler_name (str): Name of the dataset handler to use for processing.
        streaming (bool): Whether to use streaming mode for large datasets.
            When True, data is loaded on-the-fly without full memory loading.
        json_keys (list): List of keys to extract from JSON records.
            Defaults to ['text'] for single text field extraction.
        split_sentences (bool): Whether to split documents into individual sentences.
        keep_newlines (bool): Whether to preserve newline characters when splitting sentences.
        prompt_type (str): Template name for constructing prompts in training.
            "empty" means no template is applied.
        prompt_type_path (str): Path to the JSON file containing prompt templates.
            Defaults to TEMPLATES_DIR.
        dataset_additional_keys (list): Additional field keys to extract from dataset
            beyond the default keys.
        interleave_probs (str): Sampling probabilities for multiple datasets.
            Format: comma-separated values summing to 1.0.
            Example: "0.1,0.2,0.3,0.4".
        overwrite_cache (bool): Whether to overwrite existing cached training and evaluation sets.
        seed (int): Random seed for deterministic data mixing and shuffling.
        cache_dir (str): Local directory path for storing cached dataset files.
        map_keys (dict): Field name mapping for dataset column renaming.
            Format: {"original_name": "new_name"}.
        pack (bool): Whether to pack multiple samples into a single sample
            for fine-tuning dataset efficiency.
        neat_pack (bool): Whether to use zigzag attention mask for packed sequences.
        script_data_dir (str): Directory path for Python script-based dataset loading.
        tokenizer_type (str): Tokenizer implementation type.
            Default: 'HuggingFaceTokenizer'.
        tokenizer_not_use_fast (bool): Whether to disable HuggingFace fast tokenizer.
            When True, uses the slower Python implementation.
        vocab_file (str): Path to vocabulary file for custom tokenizers.
        merge_file (str): Path to BPE merge file for BPE-based tokenizers.
        append_eod (bool): Whether to append <eod> (end-of-document) token
            at the end of each document.
        tokenizer_name_or_path (str): HuggingFace tokenizer name or local path.
        seq_length (int): Maximum sequence length for model input processing.
        make_vocab_size_divisible_by (int): Padding value to make vocabulary size
            divisible by this number for computational efficiency.
        pad_vocab_size_to (int): Target vocabulary size after padding.
            Must be greater than initial tokenizer vocabulary size.
            When set, overrides `make_vocab_size_divisible_by`.
        placeholder_token (str): Special token marking step boundaries for PRM (Process Reward Model)
            predictions. Default is Cyrillic "ки".
        reward_tokens (list): Token labels indicating correctness of each reasoning step
            in the reasoning process.
        output_prefix (str): Output file path prefix (without suffix) for binary dataset files.
        dataset_impl (str): Dataset storage backend implementation.
            Options: 'lazy', 'cached', 'mmap'.
        workers (int): Number of parallel worker processes for data processing.
        n_subs (int): Number of subsets to split data for multiprocessing.
        log_interval (int): Step interval between progress logging updates.
        merge_group_keys (list): List of keys for grouping files to merge.
            Files with matching keys in 'bin-idx' filenames are merged together.
        enable_thinking (bool): Whether to enable thinking label for Qwen3 template.
    """

    def __init__(self, config_dict):
        """Initialize DataHandlerConfig with configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing the configuration parameters.
                If None, default values will be used for all attributes.
        """
        self.input = None
        self.handler_name = ""
        self.streaming = False
        self.json_keys = ['text']
        self.split_sentences = False
        self.keep_newlines = False
        self.prompt_type = "empty"
        self.prompt_type_path = TEMPLATES_DIR
        self.dataset_additional_keys = []
        self.interleave_probs = None
        self.overwrite_cache = False
        self.seed = 1234
        self.cache_dir = os.path.join(os.path.expanduser("~"), "cache")
        self.map_keys = None
        self.pack = False
        self.neat_pack = False
        self.script_data_dir = None
        self.tokenizer_type = 'HuggingFaceTokenizer'
        self.tokenizer_not_use_fast = True
        self.vocab_file = None
        self.merge_file = None
        self.append_eod = False
        self.tokenizer_name_or_path = None
        self.seq_length = None
        self.make_vocab_size_divisible_by = 128
        self.pad_vocab_size_to = None
        self.placeholder_token = "ки"
        self.reward_tokens = []
        self.output_prefix = None
        self.dataset_impl = "mmap"
        self.workers = 1
        self.n_subs = 1
        self.log_interval = 100
        self.merge_group_keys = None
        self.enable_thinking = False

        if config_dict is not None:
            self.update(config_dict)

        if self.input is None:
            raise ValueError("input is required.")

        if self.output_prefix is None:
            raise ValueError("output_prefix is required.")