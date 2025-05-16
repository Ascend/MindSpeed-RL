# coding=utf-8
# Copyright (c) 2020; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Any, Tuple

import numpy as np

from mindspeed_rl.datasets.indexed_dataset import get_packed_indexed_dataset
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers('build_train_valid_test_datasets')


def build_train_valid_test_datasets(
        data_prefix: str,
        splits_string: str,
        seq_length: int,
        train_valid_test_num_samples: Tuple,
        dataset_cls: Any,
        tokenizer: Any = None,
        parallel_state: Any = None,
        full_shuffle_instruction_dataset: bool = False,
        no_shuffle: bool = False,
        reset_position_ids: bool = False,
        prompt_type: str = None,
        prompt_type_path: str = None,
        seed: int = 42,
        extra_param: Optional[Any] = None
):
    """
    Build train, valid, and test datasets.

    Args:
        data_prefix: path+prefix for data
        splits_string: split for train,valid,test data, i.e. 80,10,10
        seq_length: sequence length used for training
        train_valid_test_num_samples: a triplet for samples, i.e. (3840, 5120, 1280)
        dataset_cls: An class implemented based on BaseDataset
        tokenizer: tokenizer by get_tokenizer
        parallel_state: megatron parallel state
        full_shuffle_instruction_dataset: full shuffle for all index
        no_shuffle: do not use shuffle index
        reset_position_ids: support for TND Training
        prompt_type: for instruction training, model related
        prompt_type_path: the path to templates.json
        seed: random seed
        extra_param: param for dataset
    """

    logger.info(' > datasets target sizes (minimum size):')
    logger.info('    train:      {}'.format(train_valid_test_num_samples[0]))
    logger.info('    validation: {}'.format(train_valid_test_num_samples[1]))
    logger.info('    test:       {}'.format(train_valid_test_num_samples[2]))


    # Only Support Single dataset.
    all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
        data_prefix=data_prefix,
        splits_string=splits_string,
        seq_length=seq_length,
        train_valid_test_num_samples=train_valid_test_num_samples,
        tokenizer=tokenizer,
        dataset_cls=dataset_cls,
        parallel_state=parallel_state,
        full_shuffle_instruction_dataset=full_shuffle_instruction_dataset,
        reset_position_ids=reset_position_ids,
        no_shuffle=no_shuffle,
        prompt_type=prompt_type,
        prompt_type_path=prompt_type_path,
        seed=seed,
        extra_param=extra_param
    )

    return all_train_datasets, all_valid_datasets, all_test_datasets


def _build_train_valid_test_datasets(
        data_prefix,
        splits_string,
        seq_length: int,
        train_valid_test_num_samples,
        tokenizer=None,
        dataset_cls=None,
        parallel_state=None,
        full_shuffle_instruction_dataset=None,
        no_shuffle=False,
        reset_position_ids=None,
        prompt_type=None,
        prompt_type_path=None,
        seed=None,
        extra_param=None
):
    """Build train, valid, and test datasets."""

    # 设置默认数据集类，保持向后兼容
    if dataset_cls is None:
        raise ValueError("dataset_cls must be provided.")

    if isinstance(data_prefix, list):
        data_prefix = data_prefix[0]

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix)

    total_num_of_documents = len(list(packed_indexed_dataset.datasets.values())[0])
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    logger.info(' > dataset split:')

    logger.info("     train:   document indices in [{}, {}) total"
                " of {} documents".format(splits[0], splits[1], splits[1] - splits[0]))
    logger.info("     validation:   document indices in [{}, {}) total"
                " of {} documents".format(splits[1], splits[2], splits[2] - splits[1]))
    logger.info("     test:   document indices in [{}, {}) total"
                " of {} documents".format(splits[2], splits[3], splits[3] - splits[2]))

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], dtype=np.int32)
            # 使用传入的dataset_cls动态创建数据集实例
            dataset = dataset_cls(
                parallel_state=parallel_state,
                dataset_type='LLM',
                data_prefix=data_prefix,
                is_packed_data=True,
                tokenizer=tokenizer,
                seq_length=seq_length,
                num_samples=train_valid_test_num_samples[index],
                name=name,
                documents=documents,
                seed=seed,
                full_shuffle_instruction_dataset=full_shuffle_instruction_dataset,
                no_shuffle=no_shuffle,
                reset_position_ids=reset_position_ids,
                prompt_type=prompt_type,
                prompt_type_path=prompt_type_path,
                extra_param=extra_param
            )

        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return train_dataset, valid_dataset, test_dataset


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)

    if splits_sum <= 0.0:
        raise ValueError("splits_num {} should be lager than 0".format(splits_sum))

    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff

    if len(splits_index) != 4:
        raise ValueError("the length of splits_index {} should be 4".format(len(splits_index)))
    if splits_index[-1] != size:
        raise ValueError("splits_index[-1] {} and size {} are supposed to be equal".format(splits_index[-1], size))
    return splits_index
