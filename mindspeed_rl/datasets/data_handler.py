# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import time
from collections import defaultdict

import torch
import numpy as np

from mindspeed_rl.datasets import indexed_dataset
from mindspeed_rl.datasets.handler_utils import greedy_knapsack
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

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])

        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        logger.info("Time to startup:%s", startup_end - startup_start)

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
                logger.info("Processed %s documents (%s docs/s).", k, self.args.log_interval / elapsed)

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

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])
        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logger.info("Time to startup:%s", startup_end - startup_start)

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
                logger.info("Processed %s documents (%s docs/s, %s MB/s).", batch_id, batch_id / elapsed, mbs)

        logger.info("Skip %s sample exceeded seq-length(%s)", skip_num / len(self.args.json_keys), self.args.seq_length)
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
