# coding=utf-8
# Copyright (c) 2020; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
import random

import torch
import numpy as np


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples(
        train_samples, train_iters, global_batch_size, eval_interval, eval_iters):
    """Train/valid/test num samples."""

    # Number of train/valid/test samples.
    if train_samples:
        train_samples = train_samples
    else:
        train_samples = train_iters * global_batch_size
    eval_iters = (train_iters // eval_interval + 1) * eval_iters
    test_iters = eval_iters

    return (
        train_samples,
        eval_iters * global_batch_size,
        test_iters * global_batch_size,
    )


def build_data_iter(dataloader, dataloader_type):
    # Build iterators.
    dl_type = dataloader_type
    if dl_type not in ['single', 'cyclic', 'external']:
        raise ValueError('dl_type should be one of (single, cyclic, external)')

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if dataloader is not None:
        data_iterator = _get_iterator(dl_type, dataloader)
    else:
        data_iterator = None

    return data_iterator


def get_prompt_index(labels, ignored_label):
    prompt_begin_list = []
    prompt_end_list = []
    in_group = False
    for idx, label in enumerate(labels):
        if label == ignored_label:
            if not in_group:
                prompt_begin_list.append(idx)
                in_group = True
        elif in_group:
            prompt_end_list.append(idx)
            in_group = False

    return prompt_begin_list, prompt_end_list


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int):
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    # truncate source
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    # truncate target
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _build_index_mappings(
        name,
        data_prefix,
        start_index,
        nb_documents,
        num_samples: int,
        seed,
        full_shuffle_instruction_dataset,
        parallel_state,
        no_shuffle=False
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    """

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}s'.format(seed)
    shuffle_idx_filename = _filename + f'_nb{nb_documents}' + '_decoder_packed_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank()
                                                    % torch.cuda.device_count()) == 0:
        if not os.path.isfile(shuffle_idx_filename):

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            epoch = 0
            shuffle_idx = []
            while len(shuffle_idx) < num_samples:
                new_document_ids = _build_shuffle_idx(
                    nb_documents=nb_documents,
                    start_index=start_index,
                    np_rng=np_rng,
                    no_shuffle=no_shuffle
                )
                shuffle_idx.extend(new_document_ids.tolist())
                epoch += 1

            if full_shuffle_instruction_dataset:
                random.shuffle(shuffle_idx)

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        counts = torch.cuda.LongTensor([1])
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_context_parallel_group())

    # Load mappings.
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r+')

    return shuffle_idx


def _build_sequential_idx(nb_documents: int, start_index):
    """Build the range [0, dataset_size)."""
    dtype_ = np.int64
    stop = start_index + nb_documents
    result = np.arange(start=start_index, stop=stop, step=1, dtype=dtype_)
    return result


def _build_shuffle_idx(nb_documents: int, start_index, np_rng, no_shuffle):
    """Build the range [0, dataset_size) and shuffle."""
    result = _build_sequential_idx(nb_documents, start_index)
    # in-place shuffling
    if not no_shuffle:
        np_rng.shuffle(result)
    return result