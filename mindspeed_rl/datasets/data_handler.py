# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import json
import os
import sys
import time
import glob
from collections import defaultdict
from typing import List, Dict

import torch
import numpy as np
from datasets import load_dataset

from mindspeed_rl.datasets import indexed_dataset
from mindspeed_rl.datasets.handler_utils import greedy_knapsack, get_handler_dataset_attr, align_dataset, \
    convert_token_to_id
from mindspeed_rl.datasets.templates import get_model_template
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers(name="data_handler")


class BaseDatasetHandler(object):
    """
    a base handler to tokenize or/and prompt your own dataset
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        self.args = args
        self.tokenizer = tokenizer
        self.splitter = splitter
        self.raw_datasets = raw_datasets
        self.max_seq_len = args.seq_length
        self.tokenized_dataset = None

    @property
    def _unwrapped_tokenizer(self):
        """get huggingface tokenizer"""
        return self.tokenizer.tokenizer

    def get_tokenized_data(self):
        """get tokenized(and prompted) data"""
        columns = next(iter(self.raw_datasets)).keys()
        remove_columns = list(set(columns) - set(self.args.json_keys))
        proc_kwargs = {} if self.args.streaming else {"num_proc": self.args.workers}
        return self.raw_datasets.map(self._filter, remove_columns=remove_columns, **proc_kwargs)

    def _pack_serialize_to_disk(self):
        """save idx and bin to disk"""
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Output prefix: {self.args.output_prefix}")
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])

        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        logger.info(f"Time to startup:{startup_end - startup_start}")

        valid_num = 0
        key_data_dict = {key: [] for key in self.args.json_keys}
        lengths = []
        length2indexes = defaultdict(list)
        for _, doc in enumerate(iter(self.tokenized_dataset), start=1):
            batch = doc["input_ids"]
            for idx, sample in enumerate(batch):
                length = len(sample)
                if length > self.args.seq_length:
                    logger.warning(f"Dropped lengthy example with length {length} > {self.args.seq_length}.")
                else:
                    lengths.append(length)
                    length2indexes[length].append(valid_num)
                    for key in self.args.json_keys:
                        key_data_dict[key].append(sample if key == 'input_ids' else doc[key][idx])
                    valid_num += 1

        logger.info(f"valid_num = {valid_num}, total_num = {len(self.tokenized_dataset)}, "
                    f"percentage : {valid_num / len(self.tokenized_dataset) * 100}%")

        knapsacks = greedy_knapsack(lengths, self.args.seq_length - 1)  # reserved for the padding token
        logger.info(f"new samples num : {len(knapsacks)}")
        for k, knapsack in enumerate(knapsacks):
            packed_data_dict = {key: [] for key in self.args.json_keys}

            for _, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                for key in self.args.json_keys:
                    packed_data_dict[key] += key_data_dict[key][index]

            if k % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                logger.info(f"Processed {k} documents ({self.args.log_interval / elapsed} docs/s).")

            pad_length = self.args.seq_length - len(packed_data_dict['input_ids'])
            if hasattr(self.tokenizer, "pad_token_id"):
                pad_token_id = self.tokenizer.pad_token_id
            elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "pad_token_id"):
                pad_token_id = self.tokenizer.tokenizer.pad_token_id
            else:
                raise ValueError("The pad_token_id attribute is missing for this tokenizer.")
            packed_data_dict['input_ids'] += [pad_token_id] * pad_length
            packed_data_dict['attention_mask'] += [1] * pad_length
            packed_data_dict['labels'] += [self.ignored_label] * pad_length

            for key in self.args.json_keys:
                if len(packed_data_dict[key]) != self.args.seq_length:
                    raise ValueError("The length of packed example should be identical to the seq_length.")

                sentence = torch.IntTensor(packed_data_dict[key])
                builders[key].add_item(sentence)
                builders[key].end_document()

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def _serialize_to_disk(self, iteration_batch_size=50):
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Output prefix: {self.args.output_prefix}")
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])
        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logger.info(f"Time to startup:{startup_end - startup_start}")

        skip_num = 0
        for i, doc in enumerate(self.tokenized_dataset.iter(batch_size=iteration_batch_size), start=1):
            # In post-training stage, we need to drop the data exceeded set sequence-length
            skip_indices = set()
            for key in self.args.json_keys:
                batch = [sentences for sentences in doc[key] if len(sentences) > 0]

                if len(batch) == 0:
                    continue

                for j, sentences in enumerate(batch):
                    for k, sentence in enumerate(sentences):
                        if self.args.seq_length is not None and len(sentence) >= self.args.seq_length:
                            skip_indices.add((j, k))

            for key in self.args.json_keys:
                batch = [sentences for sentences in doc[key] if len(sentences) > 0]

                if len(batch) == 0:
                    continue

                for j, sentences in enumerate(batch):
                    for k, sentence in enumerate(sentences):
                        if (j, k) in skip_indices:
                            skip_num = skip_num + 1
                            continue

                        total_bytes_processed += len(sentence) * np.int32().itemsize
                        builders[key].add_item(sentence)
                    builders[key].end_document()

            batch_id = i * iteration_batch_size
            if batch_id % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                logger.info(f"Processed {batch_id} documents ({batch_id / elapsed} docs/s, {mbs} MB/s).")

        logger.info(f"Skip {skip_num / len(self.args.json_keys)} sample exceeded seq-length({self.args.seq_length})")
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def serialize_to_disk(self, iteration_batch_size=50):
        """save idx and bin to disk"""
        if self.args.pack:
            if len(self.args.json_keys) == 1:  # PretrainHandler
                raise ValueError("Pre-training data processing does not need to be packed. "
                                 "Therefore, the --pack parameter is not required.")
            else:
                self._pack_serialize_to_disk()
        else:
            self._serialize_to_disk(iteration_batch_size=iteration_batch_size)

    def _tokenize(self, prompt):
        result = self._unwrapped_tokenizer(text=prompt)
        result["labels"] = result["input_ids"].copy()

        return result

    def _filter(self, sample):
        """prompt and tokenize"""
        return NotImplemented


class AlpacaStyleProcessRewardHandler(BaseDatasetHandler):
    """
    Handle alpaca style dataset format in process reward dataset used in PRM training
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "labels", 'attention_mask']
        self.args.output_prefix = self.args.output_prefix + "_packed"

        # set placeholder token
        self.placeholder_token_id = convert_token_to_id(args.placeholder_token, \
                                                        self._unwrapped_tokenizer)
        self.reward_tokens = args.reward_tokens

    def _filter(self, sample):
        inputs = self._unwrapped_tokenizer(sample["input"], padding=False, add_special_tokens=False)

        input_ids = inputs["input_ids"]
        label_values = sample["value"]

        if not isinstance(label_values, list):
            raise TypeError("labels should be a list of strings or numbers")
        label_tokens = []
        for label in label_values:
            if not (
                    self.reward_tokens is None or label in self.reward_tokens
            ):
                raise ValueError(f"label should be in reward tokens {self.reward_tokens}, got {label}")
            label_tokens.append(convert_token_to_id(label, self._unwrapped_tokenizer))

        labels = [-100] * len(input_ids)
        indices = [index for index, item in enumerate(input_ids) if item == self.placeholder_token_id]
        for index, indice in enumerate(indices):
            labels[indice] = label_tokens[index]

        input_token = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label_token = labels

        concatenated_ids = {
            "input_ids": [input_token],
            "attention_mask": [attention_mask],
            "labels": [label_token]
        }

        if len(input_token) != len(label_token):
            raise ValueError("length of input_token shoule be equal with length of label_token")

        return concatenated_ids


class AlpacaStylePairwiseHandler(BaseDatasetHandler):
    """
    Handle alpaca style dataset format in pairwise dataset used in RM | DPO training
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.train_on_inputs = False
        self.args.json_keys = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip())

    def _filter(self, sample):
        chosen_messages = sample["prompt"] + [sample["response"][0]]
        rejected_messages = sample["prompt"] + [sample["response"][1]]
        system = sample["system"][0]
        tools = sample["tools"][0]

        template = self.llama_factory_template
        tokenizer = self._unwrapped_tokenizer
        prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
        _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        IGNORE_INDEX = -100
        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids

        concatenated_ids = {
            "chosen_input_ids": [chosen_input_ids],
            "chosen_labels": [chosen_labels],
            "rejected_input_ids": [rejected_input_ids],
            "rejected_labels": [rejected_labels]
        }

        return concatenated_ids


class AlpacaStyleInstructionHandler(BaseDatasetHandler):
    """
    Handle LlamaFactory supported dataset format
    a Llama-factory Alpaca instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = None
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip())

    def _format_msg(self, sample):
        return sample

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        for source_ids, target_ids in self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0]
        ):
            if self.train_on_inputs:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [self.ignored_label] * (len(source_ids) - 1)
            else:
                source_mask = [self.ignored_label] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)

        model_inputs["input_ids"] = input_ids

        if input_ids[0] == 0:
            model_inputs["attention_mask"] = [1] * total_length
        else:
            model_inputs["attention_mask"] = [input_ids[0] // input_ids[0]] * total_length
        model_inputs["labels"] = labels
        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(messages, self.llama_factory_template, self.tokenizer.tokenizer)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        return tokenized_full_prompt


class R1AlpacaStyleInstructionHandler(BaseDatasetHandler):
    """
    Handle LlamaFactory supported dataset format
    a Llama-factory Alpaca instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        # self.prompter is unused in LlamaFactoryInstructionHandler
        self.prompter = None
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", *args.dataset_additional_keys]
        # use '_packed' string to mark that this is a _packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip())

    def _format_msg(self, sample):
        return sample

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": []}
        input_ids = []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        for source_ids, target_ids in self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0]
        ):
            input_ids += source_ids

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(input_ids)

        for add_key in self.args.dataset_additional_keys:
            if add_key == "labels":
                model_inputs["labels"] = self._unwrapped_tokenizer.encode(
                    example["response"][-1]["content"], padding=False, add_special_tokens=False)
            else:
                model_inputs[add_key] = self._unwrapped_tokenizer.encode(
                    example[add_key], padding=False, add_special_tokens=False)

        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(
            messages,
            self.llama_factory_template,
            self.tokenizer.tokenizer
        )

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]

        return tokenized_full_prompt


def _get_handler_cls(handler_name=None):
    """choose dataset class by dataset_name"""
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("current module not found")
    handler_cls = getattr(current_module, handler_name, None)
    if handler_cls is None:
        raise ValueError(f"{handler_name} is not supported.")
    logger.info(f"dataset will use {handler_cls.__name__} to handle dataset")
    return handler_cls


def get_dataset_handler(args, raw_dataset, tokenizer, splitter):
    """
    get a handler instance
    """
    handler = _get_handler_cls(args.handler_name)

    handler_instance = handler(args, raw_dataset, tokenizer, splitter)
    return handler_instance


def _get_data_format(files):
    """get format with largest number"""
    all_support_format = {
        'parquet': 'parquet',
        'arrow': 'arrow',
        'csv': 'csv',
        'json': 'json',
        'jsonl': 'json',
        'txt': 'text'
    }
    format_num = {}
    for file in files:
        ext = file.split('.')[-1]
        format_num[ext] = format_num.get(ext, 0) + 1
    exts_with_num = sorted(format_num.items(), key=lambda x: x[1], reverse=True)
    has_data_file = False
    for ext, _ in exts_with_num:
        if ext in all_support_format:
            has_data_file = True
            break
    return (ext, all_support_format.get(ext)) if has_data_file else (None, None)


def _has_py_script(input_name):
    if os.path.isdir(input_name):
        dir_name = os.path.basename(input_name)
        if os.path.exists(os.path.join(input_name, dir_name + '.py')):
            has_py_script = True
        else:
            has_py_script = False
    else:
        if input_name.split('.')[-1] == 'py':
            has_py_script = True
        else:
            has_py_script = False
    return has_py_script


def build_dataset(args):
    raw_datasets = None
    cache_dir = args.cache_dir
    split_flag = "train"
    load_from_local = os.path.exists(args.input)
    if load_from_local:
        if _has_py_script(args.input):
            logger.info("loading data from a local python script")
            raw_datasets = load_dataset(
                args.input,
                data_dir='./' if not args.script_data_dir else args.script_data_dir,
                split=split_flag,
                num_proc=None if args.streaming else args.workers,
                cache_dir=cache_dir,
                streaming=args.streaming,
                trust_remote_code=False
            )
        else:
            data_files = [args.input] if os.path.isfile(args.input) else \
                glob.glob(os.path.join(args.input, '*'))
            ext, data_format = _get_data_format(data_files)
            filtered_data_files = list(filter(lambda x: x.split('.')[-1] == ext, data_files))
            if filtered_data_files:
                logger.info(f"loading data from local file, format: {data_format},"
                            f" file num: {len(data_files)}")
                raw_datasets = load_dataset(
                    data_format,
                    split=split_flag,
                    data_files=filtered_data_files,
                    num_proc=None if args.streaming else args.workers,
                    cache_dir=cache_dir,
                    streaming=args.streaming,
                    trust_remote_code=False
                )
            else:
                raise Exception("unknown local data!")
    else:
        logger.info("loading data from remote huggingface")
        raw_datasets = load_dataset(
            args.input,
            split=split_flag,
            num_proc=None if args.streaming else args.workers,
            cache_dir=cache_dir,
            streaming=args.streaming,
            trust_remote_code=False
        )
    if raw_datasets is None:
        raise Exception("unknown data")

    if args.handler_name in [
        "AlpacaStyleInstructionHandler",
        "AlpacaStylePairwiseHandler",
        "R1AlpacaStyleInstructionHandler",
    ]:
        handler_dataset_attr = get_handler_dataset_attr(args.handler_name, args.dataset_additional_keys, args.map_keys,
                                                        raw_datasets)

        return align_dataset(raw_datasets, handler_dataset_attr, args)

    return raw_datasets
