# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import time
from abc import ABC
from typing import List, Dict, Union, Optional
from operator import itemgetter

import ray
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from mindspeed_rl.utils.loggers import Loggers


logger = Loggers("transfer_dock")


class TimeoutException(Exception):
    """Custom Timeout Exception"""
    def __init__(self, message="TIMEOUT: Time Sleep Too Long"):
        super().__init__(message)


class TransferDock(ABC):
    """
    TransferDock is a data structure class that serves as the base class for GRPOTransferDock,
    providing data storage and retrieval functions.
    """

    def __init__(self, max_len: int, experience_columns: Union[List[str], None],
                 timeout: Union[int, None], timeout_interval: Union[int, None]) -> None:
        """TransferDock initialize.

        Args:
            max_len: The maximum length of data that can be stored in TransferDock.
            experience_columns: Data columns in TransferDock.
        """
        super().__init__()
        self.max_len = max_len
        self.experience_columns = experience_columns if experience_columns is not None else []
        self.experience_data = {key: [None for _ in range(self.max_len)] for key in self.experience_columns}
        self.experience_data_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32) for key in self.experience_columns
        }
        self.index_dispatch_stop_signal = False
        self.timeout = timeout if timeout is not None else 300 # waite over 300s to log
        self.timeout_interval = timeout_interval if timeout_interval is not None else 5 # logger ever 5s

    def _put(
            self,
            experience_columns: List[str],
            experience: List[List[List[torch.Tensor]]],
            indexes: List[int] = None,
    ):
        """Put data into specified columns and rows.

        Args:
            experience_columns: Columns to put data in.
                ['prompts', 'attention_mask']
            experience: Data for the corresponding columns.
                [
                    [
                        [tensor([1, 1, 1, 1]), tensor([2, 2, 2, 2])],
                        [tensor([3, 3, 3, 3]), tensor(4, 4, 4, 4)]
                    ],
                    [
                        [tensor([1]), tensor([2, 2])],
                        [tensor([3, 3, 3]), tensor([4, 4, 4, 4])]
                    ]
                ]
            indexes: Rows to put data in.
                [0, 2]

        Returns: None

        """
        # If experience_columns not in TD, raise ValueError
        for experience_column in experience_columns:
            if experience_column not in self.experience_columns:
                raise ValueError(f"put experience ERROR: {experience_column} not in TD experience_column {self.experience_columns}")
        
        if indexes is not None:
            self._put_with_index(experience_columns, experience, indexes)
        else:
            self._put_without_index(experience_columns, experience)

    def _put_with_index(
            self,
            experience_columns: List[str],
            experience: List[List[List[torch.Tensor]]],
            indexes: List[int],
    ):
        """Put data into specified columns and rows.

        Args:
            experience_columns: Columns to put data in.
            experience: Data for the corresponding columns.
            indexes: Rows to put data in.

        Returns: None

        """
        if max(indexes) >= self.max_len:
            raise ValueError(f"Put experience index {max(indexes)} exceeds the Transfer Dock range {self.max_len}.")

        for column_idx, single_column in enumerate(experience_columns):
            for i, index in enumerate(indexes):
                # input can rewrite experience data in TD
                self.experience_data[single_column][index] = experience[column_idx][i]
                self.experience_data_status[single_column][index] = 1

    def _put_without_index(self, experience_columns: List[str], experience: List[List[List[torch.Tensor]]]):
        """Put data into specified columns and random rows.

        Args:
            experience_columns: Columns to put data in.
            experience: Data for the corresponding columns.

        Returns: None

        """
        start_time = time.time()
        while self.index_dispatch_stop_signal:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout and elapsed_time % self.timeout_interval < 0.1: # 每隔2s打印一次
                logger.warning(f"TIMEOUT: index_dispatch_stop_signal has slept {elapsed_time} second")
            # put_without_index to the TD. Only one process can sample the index at a time. The others are waiting.
            time.sleep(0.1)
        
        self.index_dispatch_stop_signal = True
        writable_indexes = (
            torch.all(
                torch.stack(
                    [self.experience_data_status[single_column] == 0 for single_column in experience_columns],
                    dim=0,
                ),
                dim=0,
                keepdim=True,
            )
            .reshape(-1)
            .nonzero(as_tuple=True)[0]
        )
        sampled_indexes_idx = torch.multinomial(
            torch.ones(len(writable_indexes)), len(experience[0]), replacement=False
        ).tolist()
        sampled_indexes = [int(writable_indexes[i]) for i in sampled_indexes_idx]

        for single_column in experience_columns:
            self.experience_data_status[single_column][sampled_indexes] = -1
        self.index_dispatch_stop_signal = False
        self._put_with_index(experience_columns, experience, sampled_indexes)

    def _get(self, experience_columns: List[str], indexes: List[int]):
        """Get data based on row and column numbers.

        Args:
            experience_columns: Columns from which to get data.
                ['prompts', 'attention_mask']
            indexes: Rows to get data from.
                [0, 2]

        Returns: Data list.
            [
                [
                    [tensor([1, 1, 1, 1]), tensor([2, 2, 2, 2])],
                    [tensor([3, 3, 3, 3]), tensor(4, 4, 4, 4)]
                ],
                [
                    [tensor([1]), tensor([2, 2])],
                    [tensor([3, 3, 3]), tensor([4, 4, 4, 4])]
                ]
            ]

        """
        if max(indexes) >= self.max_len:
            raise ValueError(f"Get experience index {max(indexes)} exceeds the Transfer Dock range {self.max_len}.")

        experience = []
        for single_column in experience_columns:
            self._wait_for_data(single_column, indexes)
            if len(indexes) == 1:
                experience.append([self.experience_data[single_column][indexes[0]]])
            else:
                experience.append(list(itemgetter(*indexes)(self.experience_data[single_column])))

        return experience

    def _wait_for_data(self, single_column: str, indexes: List[int]):
        """ Wait for data in which column and row to be ready.

        Args:
            single_column: Column that need to wait for data to be ready.
            indexes: Rows that need to wait for data to be ready.

        Returns: None

        """
        if len(indexes) == 1:
            data_ready = self.experience_data_status[single_column][indexes] == 1
        else:
            data_ready = sum(itemgetter(*indexes)(self.experience_data_status[single_column])) == len(indexes)
        
        start_time = time.time()
        while not data_ready:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout and elapsed_time % self.timeout_interval < 0.1:
                logger.warning(f"TIMEOUT: data_ready has slept {elapsed_time} second")
            # Wait until the data in a single column is ready.
            time.sleep(0.1)
            if len(indexes) == 1:
                data_ready = self.experience_data_status[single_column][indexes] == 1
            else:
                data_ready = sum(itemgetter(*indexes)(self.experience_data_status[single_column])) == len(indexes)

    def _clear_experience_data_and_status(self):
        """Clear data and data status in TransferDock.

        Returns: None

        """
        self.experience_data = {key: [None for _ in range(self.max_len)] for key in self.experience_columns}
        self.experience_data_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32) for key in self.experience_columns
        }

    def get_experience_data(self):
        """Get all data in TransferDock.

        Returns: Data dict.

        """
        return self.experience_data

    def get_experience_status(self):
        """Get all data status in TransferDock.

        Returns: Data status dict.

        """
        return self.experience_data_status

    def get_experience_len(self):
        """Get the maximum length of data in TransferDock.

        Returns: The maximum length of data.

        """
        return self.max_len


@ray.remote(max_concurrency=100, num_cpus=10)
class GRPOTransferDock(TransferDock):
    """
    GRPOTransferDock is based on TransferDock and supports managing data transfer between
    GRPO asynchronous tasks in the Ray cluster.
    """

    def __init__(self, max_len: int, metrics=None, addition_columns: Union[List[str], None] = None, 
                 addition_consumers: Union[List[str], None] = None, timeout: Union[int, None] = None,
                 timeout_interval: Union[int, None] = None) -> None:
        """GRPOTransferDock initialize.

        Args:
            max_len: The maximum length of data that can be stored in GRPOTransferDock.
        """
        self.experience_columns = [
            "prompts",
            'prompt_length',
            'responses',
            'response_length',
            "attention_mask",
            "labels",
            "input_ids",
            "actor_rollout",
            "rm_scores",
            "token_level_rewards",
            "old_log_prob",
            "ref_log_prob",
            'advantages',
            'returns'
        ]
        self.max_len = max_len
        self.experience_consumers = [
            "trainer",
            "actor_rollout",
            "actor_log_prob",
            "ref_log_prob",
            "actor_train",
            "compute_advantage",
            "rule_reward",
            "reward_scores",
            "grpo_metrics",
        ]
        # Initialize to add additional experience columns
        if addition_columns:
            for column in addition_columns:
                if column not in self.experience_columns:
                    self.experience_columns.append(column)

        # Initialize to add additional experience consumers
        if addition_consumers:
            for consumer in addition_consumers:
                if consumer not in self.experience_consumers:
                    self.experience_consumers.append(consumer)

        self.experience_consumer_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32) for key in self.experience_consumers
        }
        self.consumer_sampling_signal = {key: False for key in self.experience_consumers}
        self.metrics = metrics
        super().__init__(self.max_len, self.experience_columns, timeout, timeout_interval)

    def get_metrics(self):
        return self.metrics

    def update_metrics(self, key="", value=None):
        self.metrics.update(key, value)

    def get_experience(
            self,
            consumer: str,
            experience_columns: List[str],
            experience_count: int = None,
            indexes: List[int] = None,
            pad_id: int = None,
            multiple: int = 1,
    ):
        """Get padded experience data from GRPOTransferDock.

        Args:
            consumer: GRPO task stage to get in.
            experience_columns: Columns from which to get data.
            experience_count: Number of data to get.
            indexes: Rows from which to get data.
            pad_id: Pad token.
            multiple: The multiple of TP to pad.

        Returns: Data dict and row numbers.

        """
        # If consumer not in TD, raise ValueError
        if consumer not in self.experience_consumers:
            raise ValueError(f"get experience ERROR: {consumer} not in TD experience_consumers {self.experience_consumers}")

        # If experience_columns not in TD, raise ValueError
        for experience_column in experience_columns:
            if experience_column not in self.experience_columns:
                raise ValueError(f"get experience ERROR: {experience_column} not in TD experience_column {self.experience_columns}")
        
        if indexes is None:
            # If experience_count > the number of TD (self.max_len), raise ValueError
            if experience_count > self.max_len:
                raise ValueError(f"max_len:{self.max_len} need >= experience_count: {experience_count}")

            # If max_len is not divisible by experience_count, raise ValueError
            if self.max_len % experience_count != 0:
                raise ValueError(f"max_len:{self.max_len} need be divisible by experience_count: {experience_count}")
                
            indexes = self._sample_ready_index(consumer, experience_count, experience_columns)
            if not indexes:
                return None, None
            experience = self._get(experience_columns, indexes)
        else:
            self.experience_consumer_status[consumer][indexes] = 1
            experience = self._get(experience_columns, indexes)

        experience_batch = trans_experience_to_output(experience, experience_columns, pad_id, multiple)
        return experience_batch, indexes

    def put_experience(
            self,
            data_dict: Dict[str, Union[Tensor, List[Tensor]]],
            indexes: List[int] = None,
            num_responses: int = 1,
    ):
        """Put data into specified columns and rows.

        Args:
            data_dict: Data dict to put in GRPOTransferDock.
            indexes: Rows to put data in.
            num_responses: The number of data to put in each row.

        Returns: None

        """
        experience_columns, experience = trans_input_to_experience(data_dict, num_responses)
        self._put(experience_columns, experience, indexes)

    def _sample_ready_index(self, consumer: str, mbs: int, experience_columns: List[str]) -> Optional[List[int]]:
        """Wait for random sampling rows to be ready.

        Args:
            consumer: GRPO task stage to sample in.
            mbs: Number for rows to sample.
            experience_columns: Columns from which to sample.

        Returns: Sampled row numbers.

        """

        start_time = time.time()
        while self.consumer_sampling_signal[consumer]:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout and elapsed_time % self.timeout_interval < 0.1:
                logger.warning(f"TIMEOUT: consumer_sampling_signal has slept {elapsed_time} second")
            # only one process can sampele index. The others are waiting.
            time.sleep(0.1)

        self.consumer_sampling_signal[consumer] = True
        not_consumed_indexes = self.experience_consumer_status[consumer] == 0
        data_ready_indexes = torch.all(
            torch.stack([self.experience_data_status[single_column] == 1 for single_column in experience_columns]),
            dim=0,
        )
        usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]

        if len(usable_indexes) < mbs:
            self.consumer_sampling_signal[consumer] = False
            return None

        sampled_indexes_idx = torch.multinomial(torch.ones(len(usable_indexes)), mbs, replacement=False).tolist()
        sampled_indexes = [int(usable_indexes[i]) for i in sampled_indexes_idx]
        self.experience_consumer_status[consumer][sampled_indexes] = 1
        self.consumer_sampling_signal[consumer] = False

        return sampled_indexes

    def all_consumed(self, consumer: str):
        """If consumer has consumed all data in GRPOTransferDock.

        Args:
            consumer: GRPO task stage to consume in.

        Returns: True or False.

        """
        return self.experience_consumer_status[consumer].sum() == self.max_len

    def clear(self):
        """Reset consumer status.Clear data and data status in GRPOTransferDock.

        Returns: None

        """
        self.experience_consumer_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32) for key in self.experience_consumers
        }
        self._clear_experience_data_and_status()

    def get_consumer_status(self):
        """Get consumer status.

        Returns: Consumer status dict.

        """
        return self.experience_consumer_status


def trans_experience_to_output(
        experience: List[List[List[Tensor]]],
        experience_columns: List[str],
        pad_id: int,
        multiple: int,
):
    """Merge and pad data into dict.

    Args:
        experience: Data list.
            [
                [
                    [tensor([1, 1, 1, 1]), tensor([2, 2, 2, 2])],
                    [tensor([3, 3, 3, 3]), tensor(4, 4, 4, 4)]
                ],
                [
                    [tensor([1]), tensor([2, 2])],
                    [tensor([3, 3, 3]), tensor([4, 4, 4, 4])]
                ]
            ]
        experience_columns: Columns for the corresponding data.
            ['prompts', 'attention_mask']
        pad_id: Pad token.
            0.0
        multiple: The multiple of TP to pad.
            1

    Returns: Merged and padded data dict.
        {
            "prompts": tensor(
                [[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4]]),
            "attention_mask": tensor(
                [[1, 0, 0, 0],
                [2, 2, 0, 0],
                [3, 3, 3, 0],
                [4, 4, 4, 4]]),
        }

    """
    batch = {}
    for i, experience_column in enumerate(experience_columns):
        experience_i_all = [item for sublist in experience[i] for item in sublist]
        if experience_column in ["prompt_length", "response_length"]:
            padded = torch.cat(experience_i_all).reshape(-1, 1)
        elif experience_i_all[0].is_floating_point():
            padded = pad_multiples(experience_i_all, pad_id=0.0, multiple=multiple)
        else:
            padded = pad_multiples(experience_i_all, pad_id=pad_id, multiple=multiple)

        batch[experience_column] = padded

    return batch


def trans_input_to_experience(experience_dict: Dict[str, Union[Tensor, List[Tensor]]], num_responses: int):
    """Split data dict into columns and data list.

    Args:
        experience_dict: Data dict.
            {
                "prompts": tensor(
                    [[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4]]),
                "attention_mask": [
                    tensor([1]),
                    tensor([2, 2]),
                    tensor([3, 3, 3]),
                    tensor([4, 4, 4, 4])]
            }
        num_responses: The number of data to put in each row.
            2

    Returns: Columns and data list.
        ['prompts', 'attention_mask']
        [
            [
                [tensor([1, 1, 1, 1]), tensor([2, 2, 2, 2])],
                [tensor([3, 3, 3, 3]), tensor(4, 4, 4, 4)]
            ],
            [
                [tensor([1)], tensor([2, 2])],
                [tensor([3, 3, 3]), tensor([4, 4, 4, 4])]
            ]
        ]

    """
    experience_columns = []
    experience_list = []
    for key, value in experience_dict.items():
        if value is not None:
            experience_columns.append(key)
            if len(value) % num_responses != 0:
                raise ValueError(
                    f"value must divide num_responses, but got value: {len(value)}, num_responses: {num_responses}"
                )
            if isinstance(value, Tensor):
                value = value.reshape(len(value) // num_responses, num_responses, value.shape[1])
                slices = torch.unbind(value, dim=0)
                value = [[t.detach().clone() for t in torch.unbind(t_slice, dim=0)] for t_slice in slices]
            elif isinstance(value, List):
                value = [
                    [v.detach().clone() for v in value[i: i + num_responses]]
                    for i in range(0, len(value), num_responses)
                ]
            experience_list.append(value)

    return experience_columns, experience_list


def pad_multiples(data_list: List[Tensor], pad_id: Union[float, int], multiple: int = 1) -> Tensor:
    """Pad method for data list.

    Args:
        data_list: Data list.
        pad_id: Pad token.
        multiple: The multiple of TP to pad.

    Returns: Padded tensor.

    """
    padded = pad_sequence(data_list, batch_first=True, padding_value=pad_id)
    max_len = padded.size(1)
    target_len = ((max_len + multiple - 1) // multiple) * multiple
    padded = F.pad(padded, (0, target_len - max_len), value=pad_id)
    return padded
