# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from unittest.mock import patch, MagicMock

import os
import pytest
import torch

from mindspeed_rl.workers.base_worker import BaseRayWorker, BaseWorker

from tests.test_tools.dist_test import DistributedTest


class TestBaseRayWorker(DistributedTest):
    world_size = 1

    @pytest.fixture(autouse=True)
    def setup_environment(self):
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("RANK", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        os.environ.pop("LOCAL_RANK", None)

    @pytest.fixture
    def mock_ray_context(self):
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": ["0"]}
        return mock_context

    @patch.dict(os.environ, {"MASTER_ADDR": "1234", "MASTER_PORT": "1234"})
    @patch("ray.get_runtime_context")
    def test_init_without_localhost(self, mock_get_runtime_context, mock_ray_context):
        mock_get_runtime_context.return_value = mock_ray_context
        worker = BaseRayWorker()

        assert os.environ["MASTER_ADDR"] == "1234"
        assert os.environ["MASTER_PORT"] == "1234"

    @patch("ray.get_runtime_context")
    def test_world_size(self, mock_get_runtime_context, mock_ray_context):
        mock_get_runtime_context.return_value = mock_ray_context
        worker = BaseRayWorker()
        assert worker.world_size == worker._world_size

    @patch("ray.get_runtime_context")
    def test_rank(self, mock_get_runtime_context, mock_ray_context):
        mock_get_runtime_context.return_value = mock_ray_context
        worker = BaseRayWorker()
        assert worker.rank == worker._rank

    @patch("ray.get_runtime_context")
    def test_get_master_addr_port(self, mock_get_runtime_context, mock_ray_context):
        mock_get_runtime_context.return_value = mock_ray_context
        worker = BaseRayWorker()
        addr, port = worker.get_master_addr_port()
        assert addr == worker._master_addr
        assert port == worker._master_port


class TestBaseWorker(DistributedTest):
    world_size = 1

    @pytest.fixture
    def setUp(self):
        self.megatron_config = MagicMock()
        self.megatron_config.update = MagicMock()
        self.rl_config = MagicMock()
        self.generate_config = MagicMock()
        self.model_provider = MagicMock()
        self.initialize_func = MagicMock()
        self.tokenizer = MagicMock()
        self.get_megatron_module = MagicMock()

    @patch('os.environ')
    @patch('mindspeed_rl.workers.base_worker.BaseRayWorker.__init__')
    def test_init(self, mock_BaseRayWorker, mock_os_environ, setUp):
        mock_os_environ.__setitem__.side_effect = lambda k, v: None
        mock_os_environ.get.return_value = '0'

        worker = BaseWorker(
            megatron_config=self.megatron_config,
            rl_config=self.rl_config,
            generate_config=self.generate_config,
            model_provider=self.model_provider,
            initialize_func=self.initialize_func,
            tokenizer=self.tokenizer,
            get_megatron_module=self.get_megatron_module,
        )

        mock_os_environ.__setitem__.assert_called_once_with('CUDA_DEVICE_MAX_CONNECTIONS', '1')

        mock_BaseRayWorker.assert_called_once()

    @patch('mindspeed_rl.workers.base_worker.BaseRayWorker.__init__')
    @patch('mindspeed_rl.workers.base_worker.logger.info')
    @patch('mindspeed_rl.workers.base_worker.ray.get_runtime_context')
    @patch('os.getenv')
    def test_setup_distributed_rank(self, mock_os_getenv, mock_get_runtime_context,
                                    mock_logger, mock_BaseRayWorker, setUp):
        mock_os_getenv.return_value = 1
        mock_get_runtime_context.return_value = MagicMock(return_value=MagicMock(return_value={'NPU': [1]}))

        worker = BaseWorker(
            megatron_config=self.megatron_config,
            rl_config=self.rl_config,
            generate_config=self.generate_config,
            model_provider=self.model_provider,
            initialize_func=self.initialize_func,
            tokenizer=self.tokenizer,
            get_megatron_module=self.get_megatron_module,
        )
        worker.parallel_state = MagicMock()
        worker.vocab_parallel_cross_entropy = MagicMock()
        worker.get_args = MagicMock()
        worker.get_forward_backward_func = MagicMock()
        worker.setup_distributed_rank()

        assert mock_logger.call_count == 6

    @patch('torch.cuda.current_device')
    @patch('mindspeed_rl.workers.base_worker.trans_batch_to_data_loader')
    @patch('torch.distributed.broadcast')
    @patch('mindspeed_rl.workers.base_worker.BaseRayWorker.__init__')
    @patch('mindspeed_rl.workers.base_worker.get_pipeline_model_parallel_rank')
    @patch('mindspeed_rl.workers.base_worker.get_tensor_model_parallel_rank')
    def test_dispatch_transfer_dock_data(self, mock_get_tp, mock_get_pp, mock_BaseRayWorker,
                                         mock_broadcast, mock_data_load, mock_cuda, setUp):
        mock_get_tp.return_value = 1
        mock_get_pp.return_value = 1
        mock_data_load.return_value = 1
        mock_cuda.return_value = 'cpu'

        experience_consumer_stage = 'actor_train'
        experience_colums = []
        experience_count = 1

        worker = BaseWorker(
            megatron_config=self.megatron_config,
            rl_config=self.rl_config,
            generate_config=self.generate_config,
            model_provider=self.model_provider,
            initialize_func=self.initialize_func,
            tokenizer=self.tokenizer,
            get_megatron_module=self.get_megatron_module,
        )

        worker.td = MagicMock()
        worker.parallel_state = MagicMock()

        result_1, result_2 = worker.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                experience_colums, experience_count)

        assert mock_broadcast.call_count == 2
        assert result_1 == 1
