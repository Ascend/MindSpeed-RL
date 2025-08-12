# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import time
import threading
import json
import socket
import logging
import zmq
import torch
import numpy as np

from mindspeed_rl.trainer.utils.parallel_state import (
    get_tensor_model_parallel_src_rank,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_src_rank,
    get_pipeline_model_parallel_group,
    get_context_parallel_src_rank,
    get_context_parallel_group,
)
from mindspeed_rl.utils.loggers import Loggers


logger = Loggers('zmq_communication', logger_level=logging.INFO)


TOPICS_DISPATCH_TRANSFER_DOCKER_DATA = "dispatch_transfer_docker_data"
TOPICS_ALL = [
    TOPICS_DISPATCH_TRANSFER_DOCKER_DATA
]


ZMQ_ROLE_NONE = 0
ZMQ_ROLE_SERVER = 1
ZMQ_ROLE_CLIENT = 2


TORCH_TO_NUMPY_DTYPE = {
    'torch.float32': np.float32,
    'torch.float64': np.float64,
    'torch.int32': np.int32,
    'torch.int64': np.int64,
    'torch.uint8': np.uint8,
    'torch.int8': np.int8,
    'torch.int16': np.int16,
    'torch.bool': np.bool_,
}


class ZmqServerInfo:
    '''
    Zmq server information
    Running server on tp0/pp0 in every dp domain

    global_rank: global rank id
    dp_world_size: world size in dp group
    ip_addr: server ip
    register_port: register server port
    publisher_port: publisher server port
    reliability_port: reliability server port
    use_vllm: use vllm or not
    '''
    global_rank: int
    dp_world_size: int
    ip_addr: str
    register_port: int
    publisher_port: int
    reliability_port: int
    use_vllm: bool = False


class ZmqClientInfo:
    '''
    Zmq client information
    Running client on the device which not running server

    global_rank: global rank id
    use_vllm: use vllm or not
    '''
    global_rank: int
    use_vllm: bool = False


def serialize_tensor(tensor):
    cpu_tensor = tensor.cpu()
    meta = {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "device": str(tensor.device)
    }
    return meta, cpu_tensor.numpy().tobytes()


def serialize_dict(tensor_dict):
    serialized_dict = {}
    tensor_bytes_dict = {}

    for key, tensor in tensor_dict.items():
        meta, tensor_bytes = serialize_tensor(tensor)
        serialized_dict[key] = meta
        tensor_bytes_dict[key] = tensor_bytes

    return serialized_dict, tensor_bytes_dict


def deserialize_tensor(meta, tensor_bytes):
    try:
        torch_dtype = meta["dtype"]
        if torch_dtype in TORCH_TO_NUMPY_DTYPE:
            numpy_dtype = TORCH_TO_NUMPY_DTYPE[torch_dtype]
        else:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

        np_array = np.frombuffer(tensor_bytes, dtype=numpy_dtype)
        tensor = torch.from_numpy(np_array.copy().reshape(meta["shape"]))
        return tensor

    except Exception as e:
        logger.error(f"Error deserializing tensor: {e}")
        return None


def deserialize_dict(meta_dict, tensor_bytes_list):
    tensor_dict = {}
    if len(tensor_bytes_list) != len(meta_dict):
        raise ValueError(f"Mismatch between number of tensors ({len(tensor_bytes_list)}) "
                         f"and metadata entries ({len(meta_dict)})")

    for i, (key, meta) in enumerate(meta_dict.items()):
        tensor = deserialize_tensor(meta, tensor_bytes_list[i])
        if tensor is not None:
            tensor_dict[key] = tensor

    return tensor_dict


class ZmqServer:
    def __init__(self, server_info: ZmqServerInfo, parallel_state):
        debug_info = (f"zmq server rank: {server_info.global_rank}, dp_world_size: "
                      f"{server_info.dp_world_size}, use_vllm: {server_info.use_vllm}")
        logger.debug(debug_info)
        self.server_info = server_info
        self.parallel_state = parallel_state
        self.context = zmq.Context()
        self.publisher = None
        self.register = None
        self.reliability = None
        self.msg_seq_number = 0
        self.msg_overtime_seconds = 60
        self.init_socket()

        self.clients_pubsub = []
        self.pending_rank_ack = {}
        self.ack_checkin_thread = threading.Thread(target=self.message_ack_checkin, daemon=True)
        self.ack_checkin_thread.start()

        self.broadcast_server_info()
        self.wait_all_clients()

    def __del__(self):
        if self.publisher is not None:
            self.publisher.close()
        if self.register is not None:
            self.register.close()
        if self.reliability is not None:
            self.reliability.close()

    def init_socket(self):
        try:
            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.bind(f"tcp://{self.server_info.ip_addr}:{self.server_info.publisher_port}")
            self.register = self.context.socket(zmq.REP)
            self.register.bind(f"tcp://{self.server_info.ip_addr}:{self.server_info.register_port}")
            self.reliability = self.context.socket(zmq.PULL)
            self.reliability.bind(f"tcp://{self.server_info.ip_addr}:{self.server_info.reliability_port}")
        except Exception as e:
            logger.error(f"init socket error: {e}")
        debug_info = (f"create zmq server success, ip address: {self.server_info.ip_addr}, "
                      f"publisher port: {self.server_info.publisher_port}, "
                      f"register port: {self.server_info.register_port}, "
                      f"reliability port: {self.server_info.reliability_port}")
        logger.info(debug_info)
        logger.debug(debug_info)

    def message_ack_checkin(self):
        poller = zmq.Poller()
        poller.register(self.reliability, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(1000))
            if self.reliability not in socks or socks[self.reliability] != zmq.POLLIN:
                continue

            try:
                control = json.loads(self.reliability.recv(flags=zmq.NOBLOCK))
                if control["type"] != "ACK":
                    continue

                ack_rank_id = control["rank_id"]
                msg_seq_number = control["msg_seq_number"]

                if msg_seq_number not in self.pending_rank_ack:
                    continue
                if ack_rank_id in self.pending_rank_ack[msg_seq_number]:
                    self.pending_rank_ack[msg_seq_number].remove(ack_rank_id)
                    debug_info = f"receive pubsub checkin ack, rank: {ack_rank_id}, seq_num: {msg_seq_number}"
                    logger.debug(debug_info)
            except (zmq.ZMQError, json.JSONDecodeError, KeyError):
                pass

    def reliability_send_message(self, seq_num, message):
        debug_info = (f"send begin, seq_num: {seq_num}, "
                      f"pubsub clients: {self.clients_pubsub}")
        logger.debug(debug_info)
        overtime = 1000
        time_passed = overtime + 1
        overtime_count = 0
        self.pending_rank_ack[seq_num] = set(self.clients_pubsub)
        self.publisher.send_multipart(message)

        while len(self.pending_rank_ack[seq_num]) > 0:
            time.sleep(0.001)
            time_passed += 1
            if time_passed % overtime == 0:
                overtime_count += 1
                self.publisher.send_multipart(message)
            if overtime_count > self.msg_overtime_seconds:
                raise TimeoutError(f"wait client pubsub message ack overtime, "
                                   f"pending rank: {self.pending_rank_ack}")

        del self.pending_rank_ack[seq_num]
        debug_info = (f"send end, seq_num: {seq_num}, "
                      f"pubsub clients: {self.clients_pubsub}")
        logger.debug(debug_info)

    def broadcast_server_info(self):
        server_info_json = {
            "ip_addr": self.server_info.ip_addr,
            "publisher_port": self.server_info.publisher_port,
            "register_port": self.server_info.register_port,
            "reliability_port": self.server_info.reliability_port
        }
        byte_array = np.frombuffer(json.dumps(server_info_json).encode('utf-8'), dtype=np.uint8)
        length_tensor = torch.tensor([len(byte_array)], dtype=torch.int32).cuda()
        data_tensor = torch.tensor(byte_array, dtype=torch.uint8).cuda()

        tp_group = get_tensor_model_parallel_group(self.parallel_state, self.server_info.use_vllm)
        tp_src_rank = get_tensor_model_parallel_src_rank(self.parallel_state, self.server_info.use_vllm)
        torch.distributed.broadcast(length_tensor, tp_src_rank, group=tp_group)
        if not self.server_info.use_vllm:
            cp_src_rank = get_context_parallel_src_rank(self.parallel_state, self.server_info.use_vllm)
            cp_group = get_context_parallel_group(self.parallel_state, self.server_info.use_vllm)
            torch.distributed.broadcast(length_tensor, cp_src_rank, group=cp_group)
        pp_src_rank = get_pipeline_model_parallel_src_rank(self.parallel_state, self.server_info.use_vllm)
        pp_group = get_pipeline_model_parallel_group(self.parallel_state, self.server_info.use_vllm)
        torch.distributed.broadcast(length_tensor, pp_src_rank, group=pp_group)

        torch.distributed.broadcast(data_tensor, tp_src_rank, group=tp_group)
        if not self.server_info.use_vllm:
            cp_src_rank = get_context_parallel_src_rank(self.parallel_state, self.server_info.use_vllm)
            cp_group = get_context_parallel_group(self.parallel_state, self.server_info.use_vllm)
            torch.distributed.broadcast(data_tensor, cp_src_rank, group=cp_group)
        torch.distributed.broadcast(data_tensor, pp_src_rank, group=pp_group)


    def dispatch_transfer_docker_data(self, tensor_dict_data):
        serialized_dict, tensor_bytes_dict = serialize_dict(tensor_dict_data)
        self.msg_seq_number += 1
        message_meta = {
            "meta": serialized_dict,
            "timestamp": time.time(),
            "msg_seq_number": self.msg_seq_number
        }
        self.reliability_send_message(self.msg_seq_number,
                                      [TOPICS_DISPATCH_TRANSFER_DOCKER_DATA.encode('utf-8'),
                                       json.dumps(message_meta).encode('utf-8')] +
                                      list(tensor_bytes_dict.values()))
        return tensor_dict_data

    def wait_all_clients(self):
        ready_pubsub_rank = set()
        while len(ready_pubsub_rank) < self.server_info.dp_world_size - 1:
            message = self.register.recv_json()
            if message.get("type") == "register":
                ready_pubsub_rank.add(message.get("rank"))
                self.register.send_json({"type": "register_ack", "status": "ok"})
            else:
                logger.error(f"wait client receive unknown data")
        debug_info = (f"rank: {self.server_info.global_rank}, wait all clients success, "
                      f"pubsub clients: {ready_pubsub_rank}")
        logger.debug(debug_info)
        self.clients_pubsub = sorted(ready_pubsub_rank)


class ZmqClient:
    def __init__(self, client_info: ZmqClientInfo, parallel_state):
        debug_info = f"zmq client, rank: {client_info.global_rank}, use_vllm: {client_info.use_vllm}"
        logger.debug(debug_info)
        self.client_info = client_info
        self.parallel_state = parallel_state
        self.context = zmq.Context()
        self.register = None
        self.subscriber = None
        self.reliability = None
        self.server_info = None
        self.last_msg_seq_number = 0

        self.topic_map_callback = {
            TOPICS_DISPATCH_TRANSFER_DOCKER_DATA: self.callback_transfer_docker_data
        }

        self.wait_server_broadcast()
        self.register_to_server()

    def __del__(self):
        if self.subscriber is not None:
            self.subscriber.close()
        if self.register is not None:
            self.register.close()
        if self.reliability is not None:
            self.reliability.close()

    def wait_server_broadcast(self):
        length_tensor = torch.tensor([0], dtype=torch.int32).cuda()

        tp_src_rank = get_tensor_model_parallel_src_rank(self.parallel_state, self.client_info.use_vllm)
        tp_group = get_tensor_model_parallel_group(self.parallel_state, self.client_info.use_vllm)
        pp_src_rank = get_pipeline_model_parallel_src_rank(self.parallel_state, self.client_info.use_vllm)
        pp_group = get_pipeline_model_parallel_group(self.parallel_state, self.client_info.use_vllm)

        torch.distributed.broadcast(length_tensor, tp_src_rank, group=tp_group)
        if not self.client_info.use_vllm:
            cp_src_rank = get_context_parallel_src_rank(self.parallel_state, self.client_info.use_vllm)
            cp_group = get_context_parallel_group(self.parallel_state, self.client_info.use_vllm)
            torch.distributed.broadcast(length_tensor, cp_src_rank, group=cp_group)
        torch.distributed.broadcast(length_tensor, pp_src_rank, group=pp_group)

        data_tensor = torch.zeros(length_tensor.item(), dtype=torch.uint8).cuda()
        torch.distributed.broadcast(data_tensor, tp_src_rank, tp_group)
        if not self.client_info.use_vllm:
            cp_src_rank = get_context_parallel_src_rank(self.parallel_state, self.client_info.use_vllm)
            cp_group = get_context_parallel_group(self.parallel_state, self.client_info.use_vllm)
            torch.distributed.broadcast(data_tensor, cp_src_rank, group=cp_group)
        torch.distributed.broadcast(data_tensor, pp_src_rank, group=pp_group)

        byte_array = data_tensor.cpu().numpy()
        json_str = byte_array.tobytes().decode('utf-8')
        self.server_info = json.loads(json_str)

    def register_to_server(self):
        self.register = self.context.socket(zmq.REQ)
        self.register.connect(f"tcp://{self.server_info['ip_addr']}:{self.server_info['register_port']}")

        local_ip = self.get_local_ip(self.server_info['ip_addr'])
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.server_info['ip_addr']}:{self.server_info['publisher_port']}")

        self.register.send_json(
            {
                "type": "register",
                "rank": self.client_info.global_rank,
                "client_ip": local_ip
            }
        )
        debug_info = (f"rank: {self.client_info.global_rank}, zmq register to server, "
                      f"ip address: {self.server_info['ip_addr']}, "
                      f"register_port: {self.server_info['register_port']}, "
                      f"client_ip_addr: {local_ip}")
        logger.debug(debug_info)

        response = self.register.recv_json()
        if response.get("status") != "ok":
            raise Exception(f"register to server failed: {response}, rank: {self.client_info.global_rank}")

        self.reliability = self.context.socket(zmq.PUSH)
        self.reliability.connect(f"tcp://{self.server_info['ip_addr']}:{self.server_info['reliability_port']}")
        logger.info(f"rank: {self.client_info.global_rank}, zmq register to server success, "
                    f"ip address: {self.server_info['ip_addr']}, "
                    f"publisher_port: {self.server_info['publisher_port']}, "
                    f"register_port: {self.server_info['register_port']}, "
                    f"reliability_port: {self.server_info['reliability_port']}")
        self.subscribe_topics()

    def subscribe_topics(self):
        for topic in self.topic_map_callback.keys():
            self.subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
            logger.info(f"zmq subscribe topic: {topic}")

    def wait_publisher_message(self):
        debug_info = f"rank: {self.client_info.global_rank}, zmq wait publisher message"
        logger.debug(debug_info)
        message_parts = None
        try:
            message_parts = self.subscriber.recv_multipart()
        except Exception as e:
            logger.error(f"zmq receive data error: {e}")

        topic = message_parts[0].decode('utf-8')
        message = json.loads(message_parts[1].decode('utf-8'))
        meta_dict = message["meta"]

        reply_message = {
            "type": "ACK",
            "rank_id": self.client_info.global_rank,
            "msg_seq_number": message["msg_seq_number"]
        }
        self.reliability.send(json.dumps(reply_message).encode())
        debug_info = (f"rank: {self.client_info.global_rank}, "
                      f"reply message, seq_num: {message['msg_seq_number']}")
        logger.debug(debug_info)

        if self.last_msg_seq_number == message["msg_seq_number"]:
            debug_info = (f"rank: {self.client_info.global_rank}, receive repeat data, "
                          f"seq_num: {message['msg_seq_number']}")
            logger.debug(debug_info)
            return self.wait_publisher_message()

        self.last_msg_seq_number = message["msg_seq_number"]
        if topic not in self.topic_map_callback:
            logger.error(f"unknown topic {topic}")
            return None

        return self.topic_map_callback[topic](meta_dict, message_parts)

    @staticmethod
    def callback_transfer_docker_data(meta_dict, message_parts):
        tensor_bytes_list = message_parts[2:]
        tensor_dict = deserialize_dict(meta_dict, tensor_bytes_list)
        return tensor_dict

    @staticmethod
    def get_local_ip(master_ip):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((master_ip, 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            logger.error(f"get ip address failed: {e}")
            return "127.0.0.1"
