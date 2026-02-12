# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import logging
import random
import pytest

import ray
import torch
from tensordict import TensorDict

from mindspeed_rl.trainer.utils import TransferDock, GRPOTransferDock
from mindspeed_rl.trainer.utils.data_strategy import DataStrategy
from mindspeed_rl.config_cls.rl_config import RLConfig
from tests.test_tools.dist_test import DistributedTest
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.transfer_queue.tq_mgr import TransferQueueManager
from mindspeed_rl.utils.transfer_queue.tq_client import get_transfer_queue_client


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


@pytest.fixture(scope="function")
def setup_teardown_transfer_queue(request):
    self = request.instance
    if not ray.is_initialized():
        ray.init()
    try:
        existing = ray.get_actor("TransferQueueManager")
        ray.kill(existing)
    except ValueError:
        pass
    self.prompts_num = 2
    self.n_samples_per_prompt = 1
    self.topic = f"ut_transfer_queue_{random.randint(1000, 9999)}"
    self.experience_columns = ["prompts"]
    self.experience_consumers = ["actor_rollout"]
    self.tq_mgr = TransferQueueManager.remote(nums_tq_data=1, base_port=None)
    ray.get(self.tq_mgr.init_ready.remote())
    self.tq = get_transfer_queue_client()
    self.tq.add_topic(
        prompts_num=self.prompts_num,
        n_samples_per_prompt=self.n_samples_per_prompt,
        experience_columns=self.experience_columns,
        experience_consumers=self.experience_consumers,
        metrics=Metric(),
        topic=self.topic,
    )
    self.tq.register_consumer_columns_dict({"actor_rollout": ["prompts"]}, topic=self.topic)
    yield
    try:
        self.tq.clear_topic(topic=self.topic)
        self.tq.delete_topic(topic=self.topic)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "teardown: failed to clear/delete topic",
            exc_info=exc,
        )
    try:
        ray.kill(self.tq_mgr)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "teardown: failed to kill tq_mgr",
            exc_info=exc,
        )


@pytest.mark.usefixtures("setup_teardown_transfer_queue")
class TestTransferQueue(DistributedTest):
    is_dist_test = False

    def _create_topic(self, topic, columns, consumers, max_age=1, gbs_train=0):
        self.tq.add_topic(
            prompts_num=self.prompts_num,
            n_samples_per_prompt=self.n_samples_per_prompt,
            experience_columns=columns,
            experience_consumers=consumers,
            metrics=Metric(),
            topic=topic,
            max_age=max_age,
            GBS_train=gbs_train,
        )

    def test_put_get(self):
        prompts = [torch.randn(1, 8) for _ in range(self.prompts_num)]
        put_indexes = [0, 1]
        data_dict = {"prompts": torch.stack(prompts, dim=0)}
        self.tq.put_experience(data_dict=data_dict, indexes=put_indexes, topic=self.topic)

        batch, indexes = self.tq.get_experience(
            consumer="actor_rollout",
            experience_columns=["prompts"],
            indexes=[1],
            topic=self.topic,
        )
        assert indexes == [1]
        assert torch.equal(batch["prompts"][0], prompts[1])

    def test_all_consumed(self):
        prompts = [torch.randn(1, 8) for _ in range(self.prompts_num)]
        data_dict = {"prompts": torch.stack(prompts, dim=0)}
        self.tq.put_experience(data_dict=data_dict, indexes=[0, 1], topic=self.topic)
        batch, indexes = self.tq.get_experience(
            consumer="actor_rollout",
            experience_columns=["prompts"],
            experience_count=2,
            topic=self.topic,
        )
        assert indexes == [0, 1]
        assert self.tq.all_consumed(consumer="actor_rollout", topic=self.topic, get_n_samples=True)

    def test_consumer_columns(self):
        consumer_columns = {"actor_rollout": ["prompts"]}
        self.tq.register_consumer_columns_dict(consumer_columns, topic=self.topic)
        columns = self.tq.get_columns(consumer="actor_rollout", topic=self.topic)
        assert columns == ["prompts"]

    def test_partial_rollout_allow_partial_ready(self):
        topic = f"{self.topic}_partial"
        columns = ["prompts", "responses", "response_length"]
        consumers = ["actor_rollout"]
        self._create_topic(topic, columns, consumers, max_age=2, gbs_train=1)
        self.tq.register_consumer_columns_dict({"actor_rollout": columns}, topic=topic)
        try:
            prompts = [torch.randn(1, 8)]
            responses = [torch.tensor([1, 2], dtype=torch.int32)]
            response_length = [torch.tensor([2], dtype=torch.int32)]
            self.tq.put_experience(data_dict={"prompts": torch.stack(prompts, dim=0)}, indexes=[0], topic=topic)
            self.tq.put_experience(
                data_dict={
                    "responses": torch.stack(responses, dim=0),
                    "response_length": torch.stack(response_length, dim=0),
                },
                indexes=[0],
                topic=topic,
                data_status="partial_ready",
            )
            batch, indexes = self.tq.get_experience(
                consumer="actor_rollout",
                experience_columns=["responses", "response_length"],
                experience_count=1,
                get_n_samples=False,
                allow_partial_ready_data=True,
                topic=topic,
            )
            assert indexes == [0]
            assert batch["responses"].shape[0] == 1
        finally:
            self.tq.clear_topic(topic=topic)
            self.tq.delete_topic(topic=topic)

    def test_dp_padding(self):
        topic = f"{self.topic}_dp"
        columns = ["prompts"]
        consumers = ["actor_rollout"]
        self._create_topic(topic, columns, consumers)
        self.tq.register_consumer_columns_dict({"actor_rollout": columns}, topic=topic)
        try:
            prompts = [torch.randn(1, 8)]
            self.tq.put_experience(data_dict={"prompts": torch.stack(prompts, dim=0)}, indexes=[0], topic=topic)
            batch, indexes = self.tq.get_experience(
                consumer="actor_rollout",
                experience_columns=["prompts"],
                experience_count=2,
                indexes=[0],
                dp_size=2,
                require_dp_padding=True,
                topic=topic,
            )
            assert len(indexes) == 2
            assert indexes[1] == -2
            assert batch["prompts"].shape[0] == 2
        finally:
            self.tq.clear_topic(topic=topic)
            self.tq.delete_topic(topic=topic)

    def test_metrics_accumulate(self):
        self.tq.update_metrics(key="metric/test", value=[1.0], cumulate=True, topic=self.topic)
        self.tq.update_metrics(key="metric/test", value=[2.0], cumulate=True, topic=self.topic)
        metrics = self.tq.get_metrics(topic=self.topic)
        assert metrics.metric["metric/test"] == [1.0, 2.0]

    def test_multimodal_put_get(self):
        topic = f"multimodal_{self.topic}"
        columns = ["pixel_values", "image_num", "labels"]
        consumers = ["actor_rollout"]
        self._create_topic(topic, columns, consumers)
        self.tq.register_consumer_columns_dict({"actor_rollout": columns}, topic=topic)
        try:
            pixel_values = torch.randn(2, 3)
            image_num = torch.tensor([[1], [1]], dtype=torch.int32)
            labels = ["a", "b"]
            self.tq.put_experience(
                data_dict={"pixel_values": pixel_values, "image_num": image_num, "labels": labels},
                indexes=[0, 1],
                topic=topic,
            )
            batch, indexes = self.tq.get_experience(
                consumer="actor_rollout",
                experience_columns=["pixel_values", "image_num", "labels"],
                indexes=[0, 1],
                topic=topic,
            )
            assert indexes == [0, 1]
            assert list(batch["labels"]) == labels
            assert batch["pixel_values"].shape == pixel_values.shape
            assert batch["image_num"].shape == image_num.shape
        finally:
            self.tq.clear_topic(topic=topic)
            self.tq.delete_topic(topic=topic)


@pytest.fixture(scope="function")
def setup_teardown_data_strategy(request):
    self = request.instance
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    yield
    ray.shutdown()


@pytest.mark.usefixtures("setup_teardown_data_strategy")
@pytest.mark.parametrize("strategy", ["td", "tq"])
class TestDataStrategy(DistributedTest):
    is_dist_test = False

    def test_put_get_via_data_strategy(self, strategy):
        rl_config = RLConfig({"data_strategy": strategy})
        ds = DataStrategy(rl_config)
        prompts_num = 2
        n_samples = 1
        metrics = Metric()
        ds.build_ppo(
            prompts_num=prompts_num,
            n_samples_per_prompt=n_samples,
            metrics=metrics,
            addition_columns=[],
            addition_consumers=[],
            dataset_additional_keys=[],
        )
        data_dict = {"prompts": torch.stack([torch.randn(1, 8) for _ in range(prompts_num)], dim=0)}
        ds.put_experience(data_dict, indexes=[0, 1])

        batch, indexes = ray.get(
            ds.td.get_experience.remote(
                consumer="actor_rollout",
                experience_columns=["prompts"],
                indexes=[1],
            )
        )
        assert indexes == [1]
        assert batch["prompts"].shape[0] == 1

    def test_metrics_via_data_strategy(self, strategy):
        rl_config = RLConfig({"data_strategy": strategy})
        ds = DataStrategy(rl_config)
        metrics = Metric()
        ds.build_ppo(
            prompts_num=1,
            n_samples_per_prompt=1,
            metrics=metrics,
            addition_columns=[],
            addition_consumers=[],
            dataset_additional_keys=[],
        )
        ds.update_metrics("metric/test", value=[1.0], cumulate=True)
        ds.update_metrics("metric/test", value=[2.0], cumulate=True)
        metrics_result = ds.get_metrics()
        assert metrics_result.metric["metric/test"] == [1.0, 2.0]
