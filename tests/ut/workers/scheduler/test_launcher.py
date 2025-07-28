# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from unittest.mock import MagicMock
import pytest

import mindspeed_rl
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
from mindspeed_rl.workers.reference_woker import ReferenceWorker
from mindspeed_rl.workers.reward_woker import RewardWorker
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from tests.test_tools.dist_test import DistributedTest


def stub_build_actor_rl_config_no_colocate():
    rl_config = {
        'actor_resource': {
            'num_npus': 1
        }
    }
    return RLConfig(rl_config)


def stub_build_ref_rl_config_colocate():
    rl_config = {
        'actor_resource': {
            'num_npus': 2
        }
    }
    return RLConfig(rl_config)


def stub_build_reward_rl_config_colocate():
    rl_config = {
        'actor_resource': {
            'num_npus': 2
        }
    }
    return RLConfig(rl_config)


def stub_build_generate_config():
    return GenerateConfig({})


def stub_model_provider():
    pass


def stub_initialize_megatron():
    pass


@pytest.fixture
def setup_teardown_ray_group_actor(request):
    self = request.instance

    mock_module = MagicMock(spec=['mock_parallel_state'])
    mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=("127.0.0.1", 10000))
    mindspeed_rl.workers.scheduler.launcher.get_npu_deployment = MagicMock(return_value=1)
    mindspeed_rl.workers.scheduler.launcher.acl.rt.get_device_count = MagicMock(return_value=(4, 0))
    actor_config = MegatronConfig({}, {})
    actor_rl_config = stub_build_actor_rl_config_no_colocate()
    generate_config = stub_build_generate_config()
    self.actor_worker = RayActorGroup(
        worker=ActorHybridWorker,
        placement_group=None,
        megatron_config=actor_config,
        rl_config=actor_rl_config,
        generate_config=generate_config,
        model_provider=stub_model_provider,
        initialize_func=stub_initialize_megatron,
        parallel_state=mock_module
    ).initialize()
    yield
    del self.actor_worker


@pytest.fixture
def setup_teardown_ray_group_reference(request):
    self = request.instance

    mock_module = MagicMock(spec=['mock_parallel_state'])
    mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=("127.0.0.1", 10000))
    mindspeed_rl.workers.scheduler.launcher.get_npu_deployment = MagicMock(return_value=4)
    mindspeed_rl.workers.scheduler.launcher.acl.rt.get_device_count = MagicMock(return_value=(4, 0))
    reference_config = MegatronConfig({}, {})
    reference_rl_config = stub_build_ref_rl_config_colocate()
    generate_config = stub_build_generate_config()
    self.reference_worker = RayActorGroup(
        worker=ReferenceWorker,
        placement_group=None,
        megatron_config=reference_config,
        rl_config=reference_rl_config,
        generate_config=generate_config,
        model_provider=stub_model_provider,
        initialize_func=stub_initialize_megatron,
        parallel_state=mock_module
    ).initialize()
    yield
    del self.reference_worker


@pytest.fixture
def setup_teardown_ray_group_reward(request):
    self = request.instance

    mock_module = MagicMock(spec=['mock_parallel_state'])
    mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=("127.0.0.1", 10000))
    mindspeed_rl.workers.scheduler.launcher.get_npu_deployment = MagicMock(return_value=4)
    mindspeed_rl.workers.scheduler.launcher.acl.rt.get_device_count = MagicMock(return_value=(4, 0))
    reward_config = MegatronConfig({}, {})
    reward_rl_config = stub_build_reward_rl_config_colocate()
    generate_config = stub_build_generate_config()
    self.reward_worker = RayActorGroup(
        worker=RewardWorker,
        placement_group=None,
        megatron_config=reward_config,
        rl_config=reward_rl_config,
        generate_config=generate_config,
        model_provider=stub_model_provider,
        initialize_func=stub_initialize_megatron,
        parallel_state=mock_module
    ).initialize()
    yield
    del self.reward_worker


@pytest.mark.usefixtures("setup_teardown_ray_group_actor")
class TestRayActorGroup(DistributedTest):
    is_dist_test = False

    def test_init(self):
        assert len(self.actor_worker.actor_handlers) == 1

    def test_execute_async_command(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.get_iteration = \
            MagicMock(return_value=1)
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=1)
        iteration = self.actor_worker.execute_sync_command('get_iteration')
        assert iteration == 1

    def test_get_iteration(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.get_iteration = \
            MagicMock(return_value=1)
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=1)
        iteration = self.actor_worker.get_iteration()
        assert iteration == 1

    def test_generate_sequences(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.generate_sequences = \
            MagicMock(return_value=1)
        result = self.actor_worker.generate_sequences(blocking=False)
        assert result == 1
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=0)
        result = self.actor_worker.generate_sequences(blocking=True)
        assert result == 0

    def test_compute_log_prob(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.compute_log_prob = \
            MagicMock(return_value=1)
        result = self.actor_worker.compute_log_prob(blocking=False)
        assert result == 1
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=0)
        result = self.actor_worker.compute_log_prob(blocking=True)
        assert result == 0

    def test_update(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.update = \
            MagicMock(return_value=1)
        result = self.actor_worker.update(blocking=False)
        assert result == 1
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=0)
        result = self.actor_worker.update(blocking=True)
        assert result == 0

    def test_save_checkpoint(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.save_checkpoint = \
            MagicMock(return_value=1)
        result = self.actor_worker.save_checkpoint(blocking=False)
        assert result == 1
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=0)
        result = self.actor_worker.save_checkpoint(blocking=True)
        assert result == 0

    def test_initialize(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.initialize = \
            MagicMock(return_value=1)
        result = self.actor_worker.initialize()
        assert result == 1

    def test_get_consumed_train_samples(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.get_consumed_train_samples = \
            MagicMock(return_value=1)
        result = self.actor_worker.get_consumed_train_samples(blocking=False)
        assert result == 1
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=0)
        result = self.actor_worker.get_consumed_train_samples(blocking=True)
        assert result == 0


@pytest.mark.usefixtures("setup_teardown_ray_group_reference")
class TestRayActorGroup(DistributedTest):
    is_dist_test = False

    def test_init(self):
        assert len(self.reference_worker.actor_handlers) == 2

    def test_initialize(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.initialize = \
            MagicMock(return_value=1)
        result = self.reference_worker.initialize()
        assert result == 1


@pytest.mark.usefixtures("setup_teardown_ray_group_reward")
class TestRayActorGroup(DistributedTest):
    is_dist_test = False

    def test_init(self):
        assert len(self.reward_worker.actor_handlers) == 4

    def test_compute_rm_score(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.compute_rm_score = \
            MagicMock(return_value=1)
        result = self.reward_worker.compute_rm_score(blocking=False)
        assert result == 1
        mindspeed_rl.workers.scheduler.launcher.ray.get = MagicMock(return_value=0)
        result = self.reward_worker.compute_rm_score(blocking=True)
        assert result == 1

    def test_initialize(self):
        mindspeed_rl.workers.scheduler.launcher.RayActorGroup.initialize = \
            MagicMock(return_value=1)
        result = self.reward_worker.initialize()
        assert result == 1
