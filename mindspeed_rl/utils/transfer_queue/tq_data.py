"""
Ray actor implementing a distributed experience storage shard.

Each TransferQueueShard holds one or more ExperienceTables (per topic) in memory,
exposes a ZMQ ROUTER endpoint for low-latency PUT/GET operations, and reports data
status updates to the central TransferQueueManager. It supports dynamic table
creation/removal, index-based access, and tensor serialization over ZMQ.
"""
import pickle
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import zmq
import ray
from torch import Tensor
from ray.util import get_node_ip_address

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.transfer_queue.tq_utils import deserialize_tensor_lists, serialize_tensor_lists


@dataclass()
class ExperienceTable:
    max_len: int
    global_offset: int
    experience_columns: List[str]
    data: Dict[str, List[Any]] = field(init=False)

    def __post_init__(self):
        self.data = {col: [None] * self.max_len for col in self.experience_columns}


@ray.remote
class TransferQueueShard:
    def __init__(self, shard_id: int, port: Optional[int] = None):
        """
        Initialize a data shard.
        - shard_id: ID of this shard (for logging).
        - port: TCP port to bind the ZMQ Router server on.
        """
        self.logger = Loggers(self.__class__.__name__)

        self.shard_id = shard_id
        self.tables: Dict[str, ExperienceTable] = {}

        # Set up ZMQ Router socket for this data shard
        self.zmq_context = zmq.Context.instance()
        self.router = self.zmq_context.socket(zmq.ROUTER)
        # Bind to the specified port on all interfaces
        # If no port is provided, bind to a random available port.
        # (Use ZMQ's bind_to_random_port)
        if port is None:
            chosen_port = self.router.bind_to_random_port("tcp://0.0.0.0")
            bind_addr = f"tcp://0.0.0.0:{chosen_port}"
            port = chosen_port
        else:
            bind_addr = f"tcp://0.0.0.0:{port}"
            self.router.bind(bind_addr)
        node_ip = get_node_ip_address()
        self.endpoint = f"tcp://{node_ip}:{port}"
        self.manager = ray.get_actor("TransferQueueManager")

        self.logger.info(f"TQ_DATA[{shard_id}]: ZMQ server bound at {bind_addr}, endpoint {self.endpoint}")

        # Start background thread to handle incoming ZMQ requests
        self._running = True
        self._server_thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._server_thread.start()

    def add_experience_table(self, topic: str, max_len: int, global_offset: int, experience_columns: List[str]) -> None:
        """
        Create an ExperienceTable for a topic.
        Define storage shape only, no data insertion here.
        """
        if topic in self.tables:
            raise ValueError(f"Topic '{topic}' already exists")
        self.tables[topic] = ExperienceTable(
            max_len=max_len, global_offset=global_offset, experience_columns=experience_columns
        )

    def remove_experience_table(self, topic: str) -> None:
        """Remove the storage table for the given topic."""
        if topic in self.tables:
            del self.tables[topic]

    def clear_experience_table(self, topic: str, indexes: List[int] = None):
        """Clear stored experiences for a single topic in this shard."""
        # NOTE: storage shape (max_len/columns) is preserved.
        if topic in self.tables:
            tbl = self.tables[topic]
            if indexes is None:
                tbl.data = {col: [None] * tbl.max_len for col in tbl.experience_columns}
            else:
                for col in tbl.experience_columns:
                    for idx in indexes:
                        tbl.data[col][idx] = None

    def reorder_experience_table(self, topic: str, new_order: List[int]):
        """Clear stored experiences for a single topic in this shard."""
        # NOTE: storage shape (max_len/columns) is preserved.
        if topic in self.tables:
            tbl = self.tables[topic]
            for col, col_list in tbl.data.items():
                tbl.data[col] = [col_list[i] for i in new_order]

    def get_endpoint(self) -> str:
        """Return the connection endpoint (address:port) for this data shard."""
        return self.endpoint

    def _serve_loop(self):
        """Background thread method: listen for and handle incoming ZMQ messages."""
        while self._running:
            try:
                msg = self.router.recv_multipart()  # blocking wait for next message
            except zmq.ZMQError as e:
                self.logger.info(f"TQ_DATA[{self.shard_id}]: ZMQError in serve loop: {e}")
                break
            if not msg or len(msg) < 2:
                continue
            identity, command = msg[0], msg[1].decode()
            # Handle commands
            if command == "GET":
                # Expected frames: [identity, "GET", <pickle_data>]
                if len(msg) < 3:
                    self.router.send_multipart([identity, b"ERROR: Missing data for GET"])
                else:
                    payload = pickle.loads(msg[2])
                    try:
                        experience, returned_indexes = self._handle_get(
                            payload["topic"],
                            payload["experience_columns"],
                            payload["indexes"],
                        )

                        experience_bytes = serialize_tensor_lists(experience)
                        res_payload = {"experience_bytes": experience_bytes, "indexes": returned_indexes}
                        resp_pickled = pickle.dumps(res_payload)

                        self.router.send_multipart([identity, resp_pickled])
                    except Exception as e:
                        self.router.send_multipart([identity, f"ERROR: {e}".encode()])
            elif command == "PUT":
                # Expected frames: [identity, "PUT", <pickle_data>]
                if len(msg) < 3:
                    self.router.send_multipart([identity, b"ERROR: Missing data for PUT"])
                else:
                    payload = pickle.loads(msg[2])
                    try:
                        self._handle_put(
                            payload["topic"],
                            payload["experience_columns"],
                            deserialize_tensor_lists(payload["experience_bytes"]),
                            payload["indexes"],
                            payload["data_status"],
                        )
                        self.router.send_multipart([identity, b"ACK"])
                    except Exception as e:
                        self.router.send_multipart([identity, f"ERROR: {e}".encode()])
            else:
                self.router.send_multipart([identity, b"ERROR: Unknown command"])
        # End of loop

    def _handle_put(
        self,
        topic: str,
        experience_columns: List[str],
        experience: List[List[Tensor]],
        indexes: List[int],
        data_status: str,
    ) -> None:
        """Core insertion of experience data into this shard at the specified global indexes."""
        # 1. Validate indexes provided
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'")
        tbl = self.tables[topic]

        if not indexes:
            raise ValueError("No indexes provided for PUT")
        # 2. Check index range
        for idx in indexes:
            if idx < tbl.global_offset or idx >= tbl.global_offset + tbl.max_len:
                raise ValueError(f"Index {idx} is out of range for shard {self.shard_id}")
        # 3. Validate columns
        for col in experience_columns:
            if col not in tbl.experience_columns:
                raise ValueError(
                    f"put experience ERROR: {col} not in TQ experience_columns {tbl.experience_columns}"
                )
        # 4. Validate lengths match
        for col_idx, col in enumerate(experience_columns):
            if len(experience[col_idx]) != len(indexes):
                raise ValueError(
                    f"Length mismatch for column '{col}': {len(experience[col_idx])} values vs {len(indexes)} indexes"
                )
        # 5. Perform insertion (convert global to local index)
        for col_idx, col in enumerate(experience_columns):
            for val, idx in zip(experience[col_idx], indexes):
                local_idx = idx - tbl.global_offset
                tbl.data[col][local_idx] = val
        # 6. Notify manager of updated columns
        try:
            ray.get(self.manager.update_data_status.remote(topic, indexes, experience_columns, data_status))
        except Exception as e:
            self.logger.info(f"TQ_DATA[{self.shard_id}]: update_data_status failed: {e}")

    def _handle_get(
        self,
        topic: str,
        experience_columns: List[str],
        indexes: List[int]
    ) -> Tuple[List[List[Tensor]], List[int]]:
        """
        Retrieve raw experience data for the requested columns at the specified global indexes.

        Steps:
        1. Validate inputs (non-empty indexes, columns exist, indexes in range).
        2. For each column, wait for and collect each tensor.
        3. Return (experience_columns, experience, indexes) where experience is a list of per‐column lists of Tensors.
        """
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'")
        tbl = self.tables[topic]

        # 1a. Validate that we have indexes
        if not indexes:
            raise ValueError("No indexes provided for GET operation")

        # 1b. Validate that each column is known
        unknown = [col for col in experience_columns if col not in tbl.data]
        if unknown:
            raise ValueError(f"Unknown columns for shard {self.shard_id}: {unknown}")

        # 1c. Validate all indexes are in this shard’s range
        lo, hi = tbl.global_offset, tbl.global_offset + tbl.max_len
        bad = [idx for idx in indexes if idx < lo or idx >= hi]
        if bad:
            raise ValueError(
                f"Indexes out of range for shard {self.shard_id}: {bad} (expected in [{lo}, {hi}))"
            )

        # 2a. Fetch data per column
        local_idxes = [idx - tbl.global_offset for idx in indexes]  # compute once
        experience: List[List[Tensor]] = [
            [tbl.data[col][li] for li in local_idxes]
            for col in experience_columns
        ]
            
        # 2b. Ensure no slot is still None
        for col, lst in zip(experience_columns, experience):
            for g_idx, item in zip(indexes, lst):
                if item is None:
                    raise ValueError(f"Data at index {g_idx} for column '{col}' is not ready")

        # 3. Return raw lists plus indexes
        return experience, indexes

    def reset_all(self):
        """Drop ALL topics/tables in this shard, equivalent to post-__init__ with no topics."""
        self.tables = {}
        self.logger.info(f"TQ_DATA[{self.shard_id}]: Dropped ALL topics/tables.")

    def shutdown(self):
        """Stop the ZMQ server and background thread for this shard."""
        self._running = False
        try:
            self.router.close()
        except Exception as e:
            self.logger.error(f"TQ_DATA[{self.shard_id}]: Error closing router: {e}")
        # Wait briefly for the server thread to exit
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1)
        self.logger.info(f"TQ_DATA[{self.shard_id}]: Shutdown complete.")


    def get_values(self, topic: str, column: str, global_indexes: List[int]) -> List[Tensor]:
        """Return stored values for the given column at the specified global indexes."""
        tbl = self.tables[topic]
        lo = tbl.global_offset
        values = []
        for gi in global_indexes:
            li = gi - lo
            values.append(tbl.data[column][li])
        return values