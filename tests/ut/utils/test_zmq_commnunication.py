# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import threading
import logging
import time
import json

from unittest.mock import patch, MagicMock
import pytest
import ray
import zmq
import numpy as np
import torch

import mindspeed_rl
from mindspeed_rl.utils.zmq_communication import ZmqServer, ZmqServerInfo, ZmqClient, ZmqClientInfo
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.workers.base_worker import BaseRayWorker
from tests.test_tools.dist_test import DistributedTest


class TestZmqServer(DistributedTest):
    @pytest.fixture
    def setUp(self):
        mindspeed_rl.utils.zmq_communication.logger = Loggers('zmq_communication',
                                                              logger_level=logging.DEBUG)

    def test_init(self, setUp):
        with patch("mindspeed_rl.utils.zmq_communication.ZmqServer.broadcast_server_info") as mock_broadcast, \
             patch("mindspeed_rl.utils.zmq_communication.ZmqServer.wait_all_clients") as mock_wait, \
             patch("mindspeed_rl.utils.zmq_communication.ZmqServer.message_ack_checkin") as mock_checkin:
            server_info = MagicMock()
            server_info.global_rank = 0
            server_info.dp_world_size = 1
            server_info.use_vllm = False

            parallel_state = MagicMock()
            zmq_server = ZmqServer(server_info, parallel_state)

            assert zmq_server.server_info == server_info
            assert zmq_server.parallel_state == parallel_state
            assert zmq_server.msg_seq_number == 0
            assert zmq_server.msg_overtime_seconds == 60
            assert isinstance(zmq_server.ack_checkin_thread, threading.Thread)
            assert zmq_server.ack_checkin_thread.daemon is True
            assert len(zmq_server.clients_pubsub) == 0
            assert zmq_server.publisher is not None


class TestZmqServerClient(DistributedTest):
    world_size = 2

    @pytest.fixture
    def setUp(self):
        mindspeed_rl.utils.zmq_communication.logger = Loggers('zmq_communication',
                                                              logger_level=logging.DEBUG)


    def test_build_server_client_relation(self, setUp):
        with patch("mindspeed_rl.utils.zmq_communication.ZmqServer.broadcast_server_info") as mock_server_broadcast, \
             patch("mindspeed_rl.utils.zmq_communication.ZmqClient.wait_server_broadcast") as mock_client_wait_server:                        
            def start_zmq_server(server_info):
                zmq_server = ZmqServer(server_info, MagicMock())
                assert len(zmq_server.clients_pubsub) == 1
                assert zmq_server.clients_pubsub[0] == 1
                # 发送数据到client
                data = {"size": torch.tensor(10)}
                zmq_server.dispatch_transfer_docker_data(data)

            # 创建zmq server
            server_info = ZmqServerInfo()
            server_info.global_rank = 0
            server_info.dp_world_size = self.world_size  # 仅有一个client
            server_info.ip_addr = ray._private.services.get_node_ip_address().strip("[]")
            server_info.register_port = BaseRayWorker._get_free_port()
            server_info.publisher_port = BaseRayWorker._get_free_port()
            server_info.reliability_port = BaseRayWorker._get_free_port()
            start_server = threading.Thread(target=start_zmq_server, args=(server_info,))
            start_server.start()

            # 创建zmq client
            client_info = ZmqClientInfo()
            client_info.global_rank = 1
            with patch.object(mindspeed_rl.utils.zmq_communication.ZmqClient, 'register_to_server'):
                zmq_client = ZmqClient(client_info, MagicMock())

            server_info_json = {
                "ip_addr": server_info.ip_addr,
                "publisher_port": server_info.publisher_port,
                "register_port": server_info.register_port,
                "reliability_port": server_info.reliability_port
            }
            byte_array = np.frombuffer(json.dumps(server_info_json).encode('utf-8'), dtype=np.uint8)
            json_str = byte_array.tobytes().decode('utf-8')
            zmq_client.server_info = json.loads(json_str)

            def mock_get_json_response(data: str):
                return {
                    "status": "ok",
                    "type": "register",
                    "rank": 1,
                    'msg_seq_number': 1
                }

            zmq.sugar.socket.Socket.recv_json = mock_get_json_response
            zmq.sugar.socket.Socket.send_json = MagicMock
            zmq_client.register_to_server()
            time.sleep(1)
            assert zmq_client.register is not None
            assert zmq_client.subscriber is not None
            assert zmq_client.reliability is not None
            assert zmq_client.server_info is not None
            assert zmq_client.last_msg_seq_number == 0

            # 接收client数据
            receive_tensordict = zmq_client.wait_publisher_message()
            assert zmq_client.last_msg_seq_number == 1
            assert receive_tensordict["size"].item() == 10  # 读取到server发送过来的tensordict数据正确