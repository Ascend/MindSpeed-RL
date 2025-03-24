# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import random
import pytest

import ray
import torch

from mindspeed_rl.trainer.utils import TransferDock, GRPOTransferDock
from tests.test_tools.dist_test import DistributedTest
from mindspeed_rl.utils.metrics import Metric


@pytest.fixture(scope="function")
def setup_teardown_transfer_dock(request):
    self = request.instance
    self.max_len = 16
    self.timeout = 10
    self.timeout_interval = 2
    self.default_experience_columns = ["default"]
    self.td = TransferDock(max_len=self.max_len,
                           experience_columns=self.default_experience_columns,
                           timeout=self.timeout,
                           timeout_interval=self.timeout_interval)
    yield
    del self.td


@pytest.fixture(scope="function")
def setup_teardown_grpo_transfer_dock_function(request):
    self = request.instance
    self.max_len = 16
    metrics = Metric()
    self.td = GRPOTransferDock.remote(max_len=self.max_len, metrics=metrics)
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
    def test_init(self):
        assert self.td.max_len == self.max_len
        assert self.td.timeout == self.timeout
        assert self.td.timeout_interval == self.timeout_interval
        for experience_column in self.default_experience_columns:
            assert experience_column in self.td.experience_data
            assert experience_column in self.td.experience_data_status
        assert not self.td.index_dispatch_stop_signal

    def test_put(self):
        mbs = 4
        experience_columns = ["default"]
        indexes = random.sample(range(16), mbs)
        experience = [
            [[torch.randn(1, 8)] for _ in range(mbs)],
        ]

        self.td._put(
            experience_columns=experience_columns,
            experience=experience,
            indexes=indexes,
        )

        for index_idx, index in enumerate(indexes):
            for column_idx, experience_column in enumerate(experience_columns):
                assert torch.equal(
                    self.td.experience_data[experience_column][index][0],
                    experience[column_idx][index_idx][0],
                )

    def test_put_with_index(self):
        mbs = 4
        experience_columns = ["default"]
        indexes = random.sample(range(16), mbs)
        experience = [[[torch.randn(1, 8)] for _ in range(mbs)]]

        self.td._put_with_index(
            experience_columns=experience_columns,
            experience=experience,
            indexes=indexes,
        )

        for index_idx, index in enumerate(indexes):
            for column_idx, experience_column in enumerate(experience_columns):
                assert torch.equal(
                    self.td.experience_data[experience_column][index][0],
                    experience[column_idx][index_idx][0],
                )

    def test_put_without_index(self):
        mbs = 4
        experience_columns = ["default"]
        experience = [[[torch.randn(1, 8)] for _ in range(mbs)]]

        self.td._put_without_index(experience_columns=experience_columns, experience=experience)

        indexes = torch.nonzero(self.td.experience_data_status[experience_columns[0]], as_tuple=True)[0].tolist()
        for index in indexes:
            exp_index = []
            for column_idx, experience_column in enumerate(experience_columns):
                exp_index.extend(
                    [
                        idx
                        for idx, t in enumerate(experience[column_idx])
                        if torch.equal(t[0], self.td.experience_data[experience_column][index][0])
                    ]
                )
            assert len(set(exp_index)) == 1

    def test_get(self):
        put_mbs = 4
        get_mbs = 2
        experience_columns = ["default"]
        put_indexes = random.sample(range(16), put_mbs)
        put_experience = [
            [[torch.randn(1, 8)] for _ in range(put_mbs)],
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
                    get_experience[column_index][get_index][0],
                    put_experience[column_index][put_index][0],
                )


@pytest.mark.usefixtures("setup_teardown_grpo_transfer_dock_class")
@pytest.mark.usefixtures("setup_teardown_grpo_transfer_dock_function")
class TestGRPOTransferDock(DistributedTest):
    def test_init(self):
        experience_consumer_status = ray.get(self.td.get_consumer_status.remote())

        assert "grpo_metrics" in experience_consumer_status

    def test_get_experience(self):
        put_mbs = 4
        get_mbs = 2
        prompts = [torch.randn(1, 8) for _ in range(put_mbs)]
        put_indexes = random.sample(range(16), put_mbs)
        get_indexes = random.sample(put_indexes, get_mbs)

        ray.get(self.td.put_experience.remote(data_dict={"prompts": prompts}, indexes=put_indexes, num_responses=1))

        experience_batch, output_indexes = ray.get(
            self.td.get_experience.remote(
                consumer="actor_rollout", experience_columns=["prompts"], indexes=get_indexes, pad_id=0.0
            )
        )

        assert output_indexes == get_indexes
        assert experience_batch["prompts"].size(0) == get_mbs

    def test_put_experience(self):
        put_mbs = 4
        prompts = [torch.randn(1, 8) for _ in range(put_mbs)]
        put_indexes = random.sample(range(16), put_mbs)

        ray.get(self.td.put_experience.remote(data_dict={"prompts": prompts}, indexes=put_indexes, num_responses=1))

        for index in put_indexes:
            experience_batch, output_indexes = ray.get(
                self.td.get_experience.remote(
                    consumer="actor_rollout", experience_columns=["prompts"], indexes=[index], pad_id=0.0
                )
            )
            assert output_indexes[0] == index
            assert torch.equal(experience_batch["prompts"][0], prompts[put_indexes.index(index)])

    def test_distributed_put_experience(self):
        n_actor = 4

        @ray.remote(num_cpus=1)
        def actor_put(td, n_actor, gbs):
            for i in range(gbs // n_actor):
                data = torch.randn(1, 1024)
                ray.get(td.put_experience.remote({f"prompts": data}))

        ray.get([actor_put.remote(self.td, n_actor, self.max_len) for _ in range(n_actor)])

        _, output_indexes = ray.get(
            self.td.get_experience.remote(
                consumer="actor_rollout", experience_columns=["prompts"], experience_count=self.max_len, pad_id=0.0
            )
        )
        assert len(output_indexes) == self.max_len

    def test_distributed_get_experience(self):
        n_actor = 4

        @ray.remote(num_cpus=1)
        def actor_get(td):
            while not ray.get(td.all_consumed.remote("actor_rollout")):
                ray.get(
                    td.get_experience.remote(
                        consumer=f"actor_rollout", experience_columns=["prompts"], experience_count=1, pad_id=0.0
                    )
                )

        ray.get(
            self.td.put_experience.remote(
                data_dict={f"prompts": torch.randn(self.max_len, 1024)}, indexes=[i for i in range(self.max_len)]
            )
        )

        ray.get([actor_get.remote(self.td) for _ in range(n_actor)])

        assert torch.all(ray.get(self.td.get_consumer_status.remote())["actor_rollout"])
