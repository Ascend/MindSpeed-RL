# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import random
import pytest

import ray
import torch
from tensordict import TensorDict

from mindspeed_rl.trainer.utils import TransferDock, GRPOTransferDock
from tests.test_tools.dist_test import DistributedTest
from mindspeed_rl.utils.metrics import Metric


@pytest.fixture(scope="function")
def setup_teardown_transfer_dock(request):
    self = request.instance
    self.prompts_num = 16
    self.n_samples_per_prompt = 1
    self.max_len = self.prompts_num * self.n_samples_per_prompt
    self.timeout = 10
    self.timeout_interval = 2
    self.default_experience_columns = ["default"]
    self.td = TransferDock(prompts_num=self.prompts_num,
                           n_samples_per_prompt=self.n_samples_per_prompt,
                           experience_columns=self.default_experience_columns,
                           timeout=self.timeout,
                           timeout_interval=self.timeout_interval)
    yield
    del self.td


@pytest.fixture(scope="function")
def setup_teardown_grpo_transfer_dock_function(request):
    self = request.instance
    self.prompts_num = 16
    self.n_samples_per_prompt = 1
    self.max_len = self.prompts_num * self.n_samples_per_prompt
    metrics = Metric()
    self.td = GRPOTransferDock.remote(prompts_num=self.prompts_num, 
                                      n_samples_per_prompt=self.n_samples_per_prompt,
                                      metrics=metrics)
    yield
    ray.get(self.td.clear.remote())


@pytest.fixture(scope="class")
def setup_teardown_grpo_transfer_dock_class(request):
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    yield
    ray.shutdown()


@pytest.mark.usefixtures("setup_teardown_transfer_dock")
class TestTransferDock(DistributedTest):
    is_dist_test = False

    def test_init(self):
        assert self.td.prompts_num == self.prompts_num
        assert self.td.n_samples_per_prompt == self.n_samples_per_prompt
        assert self.td.timeout == self.timeout
        assert self.td.timeout_interval == self.timeout_interval
        assert self.td.max_len == self.max_len
        for experience_column in self.default_experience_columns:
            assert experience_column in self.td.experience_data
            assert experience_column in self.td.experience_data_status

    def test_put(self):
        mbs = 4
        experience_columns = ["default"]
        indexes = random.sample(range(16), mbs)
        experience = [
            [torch.randn(1, 8) for _ in range(mbs)],
        ]

        self.td._put(
            experience_columns=experience_columns,
            experience=experience,
            indexes=indexes,
        )

        for index_idx, index in enumerate(indexes):
            for column_idx, experience_column in enumerate(experience_columns):
                assert torch.equal(
                    self.td.experience_data[experience_column][index],
                    experience[column_idx][index_idx],
                )

    def test_get(self):
        put_mbs = 4
        get_mbs = 2
        experience_columns = ["default"]
        put_indexes = random.sample(range(16), put_mbs)
        put_experience = [
            [torch.randn(1, 8) for _ in range(put_mbs)],
        ]
        self.td._put(
            experience_columns=experience_columns,
            experience=put_experience,
            indexes=put_indexes,
        )

        get_indexes = random.sample(put_indexes, get_mbs)
        get_experience = self.td._get(experience_columns, get_indexes)

        for get_index in get_indexes:
            put_index = put_indexes.index(get_index)
            get_index = get_indexes.index(get_index)
            for experience_column in experience_columns:
                column_index = experience_columns.index(experience_column)
                assert torch.equal(
                    get_experience[column_index][get_index],
                    put_experience[column_index][put_index],
                )


@pytest.mark.usefixtures("setup_teardown_grpo_transfer_dock_class")
@pytest.mark.usefixtures("setup_teardown_grpo_transfer_dock_function")
class TestGRPOTransferDock(DistributedTest):
    is_dist_test = False

    def test_init(self):
        experience_consumer_status = ray.get(self.td.get_consumer_status.remote())

        assert "grpo_metrics" in experience_consumer_status

    def test_get_experience(self):
        put_mbs = 4
        get_mbs = 2
        prompts = [torch.randn(1, 8) for _ in range(put_mbs)]

        put_indexes = random.sample(range(16), put_mbs)
        get_indexes = random.sample(put_indexes, get_mbs)
        test_experience_data = TensorDict.from_dict({"prompts": torch.stack(prompts, dim=0)})
        ray.get(self.td.put_experience.remote(data_dict=test_experience_data, indexes=put_indexes))

        experience_batch, output_indexes = ray.get(
            self.td.get_experience.remote(
                consumer="actor_rollout", experience_columns=["prompts"], indexes=get_indexes
            )
        )

        assert output_indexes == get_indexes
        assert len(experience_batch["prompts"]) == get_mbs

    def test_put_experience(self):
        put_mbs = 4
        prompts = [torch.randn(1, 8) for _ in range(put_mbs)]
        put_indexes = random.sample(range(16), put_mbs)
        test_experience_data = TensorDict.from_dict({"prompts": torch.stack(prompts, dim=0)})
        ray.get(self.td.put_experience.remote(data_dict=test_experience_data, indexes=put_indexes))

        for index in put_indexes:
            experience_batch, output_indexes = ray.get(
                self.td.get_experience.remote(
                    consumer="actor_rollout", experience_columns=["prompts"], indexes=[index]
                )
            )
            assert output_indexes[0] == index
            assert torch.equal(experience_batch["prompts"][0], prompts[put_indexes.index(index)])
