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
        self.parallel_state = MagicMock()
        self.get_model = MagicMock()
        self.get_megatron_optimizer = MagicMock()
        self.get_optimizer_param_scheduler = MagicMock()
        self.load_checkpoint = MagicMock()
        self.save_checkpoint = MagicMock()
        self.get_args = MagicMock()
        self.tokenizer = MagicMock()
        self.get_forward_backward_func = MagicMock()

    @patch('os.environ')
    @patch('mindspeed_rl.workers.base_worker.set_parallel_state')
    @patch('mindspeed_rl.workers.base_worker.BaseRayWorker.__init__')
    def test_init(self, mock_BaseRayWorker, mock_set_parallel_state, mock_os_environ, setUp):
        mock_os_environ.__setitem__.side_effect = lambda k, v: None
        mock_os_environ.get.return_value = '0'
        
        worker = BaseWorker(
            megatron_config=self.megatron_config,
            rl_config=self.rl_config,
            generate_config=self.generate_config,
            model_provider=self.model_provider,
            initialize_func=self.initialize_func,
            parallel_state=self.parallel_state,
            get_model=self.get_model,
            get_megatron_optimizer=self.get_megatron_optimizer,
            get_optimizer_param_scheduler=self.get_optimizer_param_scheduler,
            load_checkpoint=self.load_checkpoint,
            save_checkpoint=self.save_checkpoint,
            get_args=self.get_args,
            tokenizer=self.tokenizer,
            get_forward_backward_func=self.get_forward_backward_func,
        )

        mock_os_environ.__setitem__.assert_called_once_with('CUDA_DEVICE_MAX_CONNECTIONS', '1')

        mock_BaseRayWorker.assert_called_once()
        mock_set_parallel_state.assert_called_once()

        assert worker.parallel_state == self.parallel_state

        assert worker.get_model == self.get_model


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
            parallel_state=self.parallel_state,
            get_model=self.get_model,
            get_megatron_optimizer=self.get_megatron_optimizer,
            get_optimizer_param_scheduler=self.get_optimizer_param_scheduler,
            load_checkpoint=self.load_checkpoint,
            save_checkpoint=self.save_checkpoint,
            get_args=self.get_args,
            tokenizer=self.tokenizer,
            get_forward_backward_func=self.get_forward_backward_func
        )
        worker.setup_distributed_rank()

        assert mock_logger.call_count == 6

    @patch('torch.cuda.current_device')
    @patch('mindspeed_rl.workers.base_worker.trans_batch_to_data_loader')
    @patch('torch.distributed.broadcast')
    @patch('mindspeed_rl.workers.base_worker.BaseRayWorker.__init__')
    @patch('mindspeed_rl.workers.base_worker.get_pipeline_model_parallel_rank')
    @patch('mindspeed_rl.workers.base_worker.get_tensor_model_parallel_rank')
    @patch('torch.distributed.barrier')
    def test_dispatch_transfer_dock_data(self, mock_barrier, mock_get_tp, mock_get_pp, mock_BaseRayWorker,
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
            parallel_state=self.parallel_state,
            get_model=self.get_model,
            get_megatron_optimizer=self.get_megatron_optimizer,
            get_optimizer_param_scheduler=self.get_optimizer_param_scheduler,
            load_checkpoint=self.load_checkpoint,
            save_checkpoint=self.save_checkpoint,
            get_args=self.get_args,
            tokenizer=self.tokenizer,
            get_forward_backward_func=self.get_forward_backward_func
        )

        worker.td = MagicMock()
        
        result_1, result_2 = worker.dispatch_transfer_dock_data(experience_consumer_stage, 
                                                                experience_colums, experience_count)
                                                            
        assert mock_barrier.call_count == 2
        assert mock_broadcast.call_count == 2
        assert result_1 == 1


    @patch('mindspeed_rl.workers.base_worker.BaseRayWorker.__init__')
    def test_truncate_rows(self, mock_BaseRayWorker, setUp):
        tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        index_tensor = torch.tensor([[2], [3], [1]])
        expected_output = [torch.tensor([1, 2]), torch.tensor([5, 6, 7]), torch.tensor([9])]
        worker = BaseWorker(
            megatron_config=self.megatron_config,
            rl_config=self.rl_config,
            generate_config=self.generate_config,
            model_provider=self.model_provider,
            initialize_func=self.initialize_func,
            parallel_state=self.parallel_state,
            get_model=self.get_model,
            get_megatron_optimizer=self.get_megatron_optimizer,
            get_optimizer_param_scheduler=self.get_optimizer_param_scheduler,
            load_checkpoint=self.load_checkpoint,
            save_checkpoint=self.save_checkpoint,
            get_args=self.get_args,
            tokenizer=self.tokenizer,
            get_forward_backward_func=self.get_forward_backward_func
        )
        result = worker.truncate_rows(tensor, index_tensor)
        for res, expected in zip(result, expected_output):
            assert torch.allclose(res, expected, atol=1e-5)
