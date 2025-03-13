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


class TransferDock(ABC):
    """
    TransferDock is a data structure class that serves as the base class for GRPOTransferDock,
    providing data storage and retrieval functions.
    """

    def __init__(self, max_len: int, experience_columns: Union[List[str], None]) -> None:
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
        self.initialize_stop_signal = {key: False for key in self.experience_columns}
        self.index_dispatch_stop_signal = False

    def _put(
            self,
            experience_columns: List[str],
            experience: List[List[List[torch.Tensor]]],
            indexes: List[int] = None,
    ):
        """Put data into specified columns and rows.

        Args:
            experience_columns: Columns to put data in.
            experience: Data for the corresponding columns.
            indexes: Rows to put data in.

        Returns: None

        """
        for single_column in experience_columns:
            if single_column not in self.experience_data.keys():
                self.initialize_stop_signal[single_column] = True
                self.experience_columns.append(single_column)
                self.experience_data[single_column] = [None for _ in range(self.max_len)]
                self.experience_data_status[single_column] = torch.zeros(self.max_len, dtype=torch.int32)
                self.initialize_stop_signal[single_column] = False

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
        for column_idx, single_column in enumerate(experience_columns):
            if max(indexes) >= self.max_len:
                raise ValueError("请求index超过数据结构范围")

            for i, index in enumerate(indexes):
                while (
                        single_column not in self.initialize_stop_signal.keys()
                        or self.initialize_stop_signal[single_column]
                ):
                    time.sleep(0.1)
                self.experience_data[single_column][index] = experience[column_idx][i]
                self.experience_data_status[single_column][index] = 1

    def _put_without_index(self, experience_columns: List[str], experience: List[List[List[torch.Tensor]]]):
        """Put data into specified columns and random rows.

        Args:
            experience_columns: Columns to put data in.
            experience: Data for the corresponding columns.

        Returns: None

        """
        while self.index_dispatch_stop_signal:
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
            indexes: Rows to get data from.

        Returns: Data list.

        """
        max_index = max(indexes)
        if max_index >= self.max_len:
            raise ValueError("请求index超过数据结构范围")

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
        while not data_ready:
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

    def __init__(self, max_len: int) -> None:
        """GRPOTransferDock initialize.

        Args:
            max_len: The maximum length of data that can be stored in GRPOTransferDock.
        """
        self.experience_columns = [
            "prompts",
            "attention_mask",
            "labels",
            "input_ids",
            "actor_rollout",
            "rm_score",
            "token_level_rewards",
            "old_log_prob",
            "ref_log_prob",
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
        self.experience_consumer_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32) for key in self.experience_consumers
        }
        self.consumer_sampling_signal = {key: False for key in self.experience_consumers}
        super().__init__(self.max_len, self.experience_columns)

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
        if indexes is None:
            if self.max_len % experience_count != 0:
                raise ValueError(f"max_len:{self.max_len} need >= experience_count: {experience_count}")
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
        experience_column_ready = all(
            [single_column in self.experience_data.keys() for single_column in experience_columns]
        )

        while not experience_column_ready:
            time.sleep(0.1)
            experience_column_ready = all(
                [single_column in self.experience_data.keys() for single_column in experience_columns]
            )

        while self.consumer_sampling_signal[consumer]:
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
        experience_columns: Columns for the corresponding data.
        pad_id: Pad token.
        multiple: The multiple of TP to pad.

    Returns: Merged and padded data dict.

    """
    batch = {}
    for i, experience_column in enumerate(experience_columns):
        experience_i_all = [item for sublist in experience[i] for item in sublist]
        if experience_column in ["prompt_length", "response_length"]:
            padded = torch.cat(experience_i_all).reshape(-1, 1)
        elif experience_column in ["prompts", "input_ids", "labels"]:
            padded = pad_multiples(experience_i_all, pad_id=pad_id, multiple=multiple)
        elif experience_column == "responses":
            padded = pad_multiples(experience_i_all, pad_id=-100, multiple=multiple)
        else:
            padded = pad_multiples(experience_i_all, pad_id=0.0, multiple=multiple)

        batch[experience_column] = padded

    return batch


def trans_input_to_experience(experience_dict: Dict[str, Union[Tensor, List[Tensor]]], num_responses: int):
    """Split data dict into columns and data list.

    Args:
        experience_dict: Data dict.
        num_responses: The number of data to put in each row.

    Returns: Columns and data list.

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
