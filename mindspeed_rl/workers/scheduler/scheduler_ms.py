import socket
import os
import ray


class WorkerMetaMS:
    keys = [
        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES",
        "MS_WORKER_NUM", "MS_ROLE", "MS_SCHED_HOST", "MS_SCHED_PORT"
    ]

    def __init__(self, store) -> None:
        self._store = store

    def to_dict(self):
        return {f"_{key.lower()}": self._store.get(f"_{key.lower()}", None) for key in WorkerMetaMS.keys}


@ray.remote
class WorkerGroupRegisterCenterMS:

    def __init__(self, rank_zero_info):
        self.rank_zero_info = rank_zero_info

    def get_rank_zero_info(self):
        return self.rank_zero_info


def create_worker_group_register_center_ms(name, info):
    return WorkerGroupRegisterCenterMS.options(name=name).remote(info)



class WorkerMS:

    def __init__(self, cuda_visible_devices=None) -> None:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        self._rank = rank
        self._world_size = world_size

        ms_local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", world_size))
        ms_local_rank = int(os.getenv("LOCAL_RANK", rank))

        store = {}
        store['_ms_worker_num'] = world_size
        store['_local_rank'] = ms_local_rank
        store['_local_world_size'] = ms_local_world_size
        store['_rank'] = rank
        store['_world_size'] = world_size

        if cuda_visible_devices is not None:
            store['_cuda_visible_devices'] = cuda_visible_devices

        meta = WorkerMetaMS(store=store)
        self._configure_with_meta_ms(meta_ms=meta)

    def _configure_before_init_ms(self, register_center_name: str, rank: int):
        if not isinstance(rank, int):
            raise TypeError(f"rank must be int, instead of {type(rank)}")

        if rank == 0:
            ms_master_addr, ms_master_port = self.get_availale_master_addr_port_ms()

            rank_zero_info_ms = {}
            rank_zero_info_ms["MASTER_ADDR"] = ms_master_addr
            rank_zero_info_ms["MASTER_PORT"] = ms_master_port

            if os.getenv("WG_BACKEND", None) == "ray":
                self.register_center = create_worker_group_register_center_ms(name=register_center_name,
                                                                           info=rank_zero_info_ms)
            os.environ.update(rank_zero_info_ms)

    def __new__(cls, *args, **kwargs):
        instance_ms = super().__new__(cls)

        # note that here we use int to distinguish
        disable_worker_init_ms = int(os.environ.get('DISABLE_WORKER_INIT', 0))
        if disable_worker_init_ms:
            return instance_ms
        
        worker_group_prefix_ms = os.environ.get("WG_PREFIX", None)
        rank_ms = os.environ.get("RANK", None)

        if None not in [rank_ms, worker_group_prefix_ms] and 'ActorClass(' not in cls.__name__:
            instance_ms._configure_before_init_ms(f"{worker_group_prefix_ms}_register_center", int(rank_ms))

        return instance_ms

    def _configure_with_meta_ms(self, meta_ms: WorkerMetaMS):
        """
        This function should only be called inside by WorkerGroup
        """
        if not isinstance(meta_ms, WorkerMetaMS):
            raise TypeError(
                f"Invalid meta type: expected WorkerMetaMS, got {type(meta_ms).__name__}. "
                f"(Received value: {repr(meta_ms)})"
            )
        self.__dict__.update(meta_ms.to_dict())  # this is hacky
        for key in WorkerMetaMS.keys:
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = ""

    def _get_node_ip_ms(self):

        def get_node_ip_by_sdk():
            if os.getenv("WG_BACKEND", None) == "ray":
                return ray._private.services.get_node_ip_address()
            return None

        ms_host_ipv4 = os.getenv("MY_HOST_IP", None)
        ms_host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        ms_host_ip_by_env = ms_host_ipv4 or ms_host_ipv6
        ms_host_ip_by_sdk = get_node_ip_by_sdk()

        return ms_host_ip_by_env or ms_host_ip_by_sdk
    
    def _get_free_port(self):
        with socket.socket() as sock_ms:
            sock_ms.bind(('', 0))
            return sock_ms.getsockname()[1]

    def get_availale_master_addr_port_ms(self):
        return self._get_node_ip_ms(), str(self._get_free_port())

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


@ray.remote
class WorkerGroupSchedulerMS(WorkerMS):
    def __init__(self):
        super().__init__()
        self.success = False
        with socket.socket() as sock:
            sock.bind(("", 0))
            self.port_ = sock.getsockname()[1]
        self.host_ = ray._private.services.get_node_ip_address()
        rank_zero_info = {
                "MS_SCHED_HOST": str(self.host_),
                "MS_SCHED_PORT": str(self.port_),
            }
        os.environ.update(rank_zero_info)

    def init_process_group(self):
        import mindspore as ms
        from mindspore import mint
        if not ms.communication._comm_helper._is_initialized():
            mint.distributed.init_process_group(
                backend="hccl"
            )
            self.success = True
    
    def get_status(self):
        return self.success

    def _get_free_port(self):
        return self.port_

    def _get_current_node_ip(self):
        return self.host_
 
 
def create_worker_group_scheduler(name, world_size):
    env_vars: dict[str, str] = {
        "MS_ROLE": "MS_SCHED",
        "MS_WORKER_NUM": str(world_size),
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        'WORLD_SIZE': str(world_size),
        'WG_BACKEND': 'ray',
    }
    
    options = {'runtime_env': {'env_vars': env_vars}, 'name': name}
    return WorkerGroupSchedulerMS.options(**options).remote()