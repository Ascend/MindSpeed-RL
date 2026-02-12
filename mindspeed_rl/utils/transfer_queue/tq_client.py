"""
Transfer Queue Client for Distributed Experience Storage and Retrieval

This module defines TransferQueueClient, a client for interacting with a Ray-managed distributed Transfer Queue 
that efficiently stores and retrieves sharded experience data (e.g., model outputs, rewards) in large-scale RL or generative training. 

The client communicates with the manager via Ray remote calls for metadata operations, 
while data transfer to/from TQ_DATA shards is performed over lightweight ZMQ DEALER sockets for low-latency.
"""
import pickle
import time
import threading
from typing import Dict, List, Union, Optional, Tuple
import zmq
import ray

import torch
from torch import Tensor

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.transfer_queue.tq_utils import deserialize_tensor_lists, trans_data_dict_to_experience, serialize_tensor_lists, padding_experience_for_dp
from mindspeed_rl.utils.pad_process import padding_dict_to_tensor_dict, remove_padding_tensor_dict_to_dict

logger = Loggers("tq_client")


DEFAULT_TOPIC = "default_topic"


class TransferQueueClient:
    def __init__(self, manager_handle):
        """
        Initialize the client interface.
        - manager_handle: A Ray actor handle for TQ_MGR.
        """
        self.manager = manager_handle
        # Each client will have its own ZMQ context and can reuse sockets.
        self.zmq_context = zmq.Context.instance()
        self.sockets: Dict[str, zmq.Socket] = {}  # cache Dealer sockets per endpoint
        # Per-thread storage for ZMQ sockets to ensure thread-safety.
        # ZeroMQ sockets are not thread-safe; each thread must use its own sockets.
        self._local = threading.local()
        self.logger = Loggers(self.__class__.__name__)
        self._dapo_running: Optional[bool] = None

    def _get_socket(self, endpoint: str) -> zmq.Socket:
        """Get or create a Dealer socket connected to the given endpoint."""
        # Use thread-local sockets because ZeroMQ sockets are not thread-safe.
        if not hasattr(self._local, "sockets"):
            self._local.sockets = {}
        tl_sockets: Dict[str, zmq.Socket] = self._local.sockets
        if endpoint in tl_sockets:
            return tl_sockets[endpoint]
        try:
            sock = self.zmq_context.socket(zmq.DEALER)
            sock.connect(endpoint)
            tl_sockets[endpoint] = sock
            return sock
        except Exception as e:
            self.logger.error(f"Failed to create or connect socket for endpoint {endpoint}: {e}")
            if sock in tl_sockets:
                tl_sockets.pop(endpoint)
            if sock:
                sock.close()
            raise
    
    def close_socket(self, endpoint: str) -> None:
        """Close the socket for the given endpoint."""
        if hasattr(self._local, "sockets"):
            tl_sockets: Dict[str, zmq.Socket] = self._local.sockets
            if endpoint in tl_sockets:
                sock = tl_sockets.pop(endpoint)
                try:
                    sock.close()
                except Exception as e:
                    self.logger.error(f"Failed to close socket for endpoint {endpoint}: {e}")

    def close_all_sockets(self) -> None:
        """Close all thread-local sockets."""
        if hasattr(self._local, "sockets"):
            tl_sockets: Dict[str, zmq.Socket] = self._local.sockets
            for endpoint, sock in list(tl_sockets.items()):
                try:
                    sock.close()
                except Exception as e:
                    self.logger.error(f"Failed to close socket for endpoint {endpoint}: {e}")
                tl_sockets.pop(endpoint)

    def is_running_dapo(self) -> bool:
        if self._dapo_running is not None:
            return self._dapo_running
        self._dapo_running = ray.get(self.manager.is_running_dapo.remote())
        return self._dapo_running

    def add_topic(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        experience_columns: List[str],
        experience_consumers: List[str],
        timeout: float = 5.0,
        metrics: Metric = None,
        topic: str = DEFAULT_TOPIC,
        max_age: int = 1,
        GBS_train: int = 0,
    ) -> None:
        """
        Register a topic on the manager and provision per-shard storage.
        - Pure schema setup; no data is inserted here.
        - Raises if the topic already exists.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        if not topic:
            raise ValueError("Topic must be non-empty")
        ray.get(self.manager.add_topic.remote(
            topic, prompts_num, n_samples_per_prompt, experience_columns, experience_consumers, timeout=timeout, metrics=metrics, max_age=max_age, GBS_train=GBS_train
        ))
        self.logger.info(
            f"TQ_Client: Created topic '{topic}' with prompts_num={prompts_num}, "
            f"n_samples_per_prompt={n_samples_per_prompt}, "
            f"experience_columns={experience_columns}, "
            f"experience_consumers={experience_consumers}"
        )

    def delete_topic(self, topic: str = DEFAULT_TOPIC) -> None:
        """Delete the specified topic, including tables on the manager and all shards."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.delete_topic.remote(topic))
        self.logger.info(f"TQ_Client: Deleted topic '{topic}'")

    def clear_topic(self, topic: str = DEFAULT_TOPIC):
        """Clear data the specified topic across all shards, and reset its manager-side states."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.clear_topic.remote(topic))
        self.logger.info(f"TQ_Client: Cleared data for topic '{topic}'.")

    def refine_topic_for_partial_rollout(self, topic: str = DEFAULT_TOPIC):
        """Clears trained experiences and reorders remaining data for the partial rollout."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.refine_topic_for_partial_rollout.remote(topic))
        self.logger.info(f"TQ_Client: Refined data for topic '{topic}'.")

    def clear_and_resize_topic(self, prompts_num: int, n_samples_per_prompt: int, topic: str = DEFAULT_TOPIC) -> None:
        """Clear and resize the specified topic. Other parameters (column names, consumers, metrics) are preserved."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.resize_topic.remote(
            topic, prompts_num, n_samples_per_prompt
        ))
        self.logger.info(
            f"TQ_Client: Resized topic '{topic}' to prompts_num={prompts_num}, "
            f"n_samples_per_prompt={n_samples_per_prompt}"
        )

    def register_consumer_columns_dict(self, columns_dict: Dict[str, List[str]], topic=DEFAULT_TOPIC) -> None:
        ray.get(self.manager.register_consumer_columns_dict.remote(columns_dict, topic))

    def put_experience(
        self,
        data_dict: Dict[str, Union[Tensor, List[Tensor]]],
        indexes: List[int],
        unpad: bool = True,
        topic: str = DEFAULT_TOPIC,
        data_status: str = "ready",
    ) -> None:
        """
        Store experience data (e.g., model outputs, rewards) in TQ.

        Args:
            data_dict: dict from col names to either:
                - a padded Tensor of shape (batch_size, ...), or
                - a List of Tensors for each example.
            indexes: Global indexes for each example, [0, prompt_num*n_samples_per_prompt).
        """
        start_time = time.perf_counter()
        if topic is None:
            topic = DEFAULT_TOPIC
        if not indexes:
            raise ValueError("No indexes provided for put_experience")

        max_len = ray.get(self.manager.get_max_len.remote(topic))
        original_len = len(indexes)
        valid_positions = [pos for pos, idx in enumerate(indexes) if 0 <= idx < max_len]
        if len(valid_positions) != original_len:
            self.logger.warning(
                f"put_experience: drop out-of-range indexes for topic={topic}, "
                f"max_len={max_len}, min_idx={min(indexes)}, max_idx={max(indexes)}"
            )
            indexes = [indexes[pos] for pos in valid_positions]
        if not indexes:
            self.logger.warning(f"put_experience: no valid indexes for topic={topic}, skip")
            return

        # 1. Remove padding
        if(unpad):
            data_dict = remove_padding_tensor_dict_to_dict(data_dict)

        if valid_positions and len(valid_positions) != original_len:
            # keep data aligned with filtered indexes
            filtered = {}
            for key, value in data_dict.items():
                if isinstance(value, list) and len(value) >= len(valid_positions):
                    filtered[key] = [value[pos] for pos in valid_positions]
                elif isinstance(value, torch.Tensor) and value.size(0) >= len(valid_positions):
                    filtered[key] = value[valid_positions]
                else:
                    filtered[key] = value
            data_dict = filtered

        # 2. Put multimodal data "labels" into TQ
        if "labels" in data_dict.keys() and "multimodal" in topic:
            mm_labels = data_dict.pop("labels", None)
            ray.get(self.manager.put_mm_labels.remote(topic, mm_labels, indexes))

        # 3. Store shape of multi-dim tensor 
        for column, experience in data_dict.items():
            if isinstance(experience[0], torch.Tensor):
                if experience[0].ndim >= 2:
                    shape_list = [exp.shape for exp in experience]
                    data_dict[column] = [exp.flatten() for exp in experience]
                    ray.get(self.manager.store_shape.remote(topic, column, indexes, shape_list))

        # 4. Transform into column names and per-example tensors
        columns, experience_list = trans_data_dict_to_experience(data_dict)

        # 5. Determine TQ_DATA shard(s) and group indexes
        targets: Dict[str, List[int]] = ray.get(
            self.manager.get_shard_for_indexes.remote(topic, indexes)
        )

        # 6. Send each shard its subset
        #    Build a position map for slicing experience_list
        pos_map = {idx: pos for pos, idx in enumerate(indexes)}
        for endpoint, idx_subset in targets.items():
            sock = self._get_socket(endpoint)

            # slice the per-column lists for this shard's indexes
            sub_experience = [
                [col_vals[pos_map[i]] for i in idx_subset]
                for col_vals in experience_list
            ]

            # pre-serialize tensors to bytes to avoid pickle's high overhead on raw tensors
            sub_experience_bytes = serialize_tensor_lists(sub_experience)

            payload = {
                "topic": topic,
                "experience_columns": columns,
                "experience_bytes": sub_experience_bytes,
                "indexes": idx_subset,
                "data_status": data_status,
            }
            pickled_payload = pickle.dumps(payload)

            sock.send_multipart([b"PUT", pickled_payload])
            reply = sock.recv_multipart()
            if not (reply and reply[0] == b"ACK"):
                err = reply[0].decode() if reply else "No reply"
                raise RuntimeError(f"Failed to put experience to shard {endpoint}: {err}")

        total_elapsed = time.perf_counter() - start_time
        ray.get(self.manager.accumulate_timing.remote("put", float(total_elapsed)))

    def get_experience(
        self,
        consumer: str,
        experience_columns: List[str] = None,
        experience_count: int = None,
        indexes: List[int] = None,
        get_n_samples: bool = True,
        pad: bool = True,
        topic: str = DEFAULT_TOPIC,
        use_batch_seqlen_balance: bool = False,
        dp_size: int = 1,
        require_dp_padding: bool = False,
        allow_partial_ready_data: bool = False,
    ) -> Tuple[Dict[str, Union[Tensor, List[Tensor]]], List[int]]:
        """
        Retrieve experience data from the distributed Transfer Queue.

        Steps:
        1. Disallow explicit indexes (must be None for sampling).
        2. Ask the manager to allocate shard(s) and their indexes.
        3. For each shard endpoint, send a GET request over ZMQ with timeout.
        4. Collect raw lists of 1D Tensors from all shards.
        5. Optionally pad and return with global indexes.

        Returns:
            - data_batch: dict from col names to either
                * a padded 2D Tensor (if pad=True), or
                * a list of 1D Tensors (if pad=False)
            - returned_indexes: list of all global indexes fetched
        """
        start_time = time.perf_counter()
        if topic is None:
            topic = DEFAULT_TOPIC

        if not experience_columns:
            experience_columns = ray.get(self.manager.get_columns.remote(topic, consumer))
            if not experience_columns:
                raise ValueError(
                    f"No experience_columns provided, and no experience columns registered for consumer "
                    f"'{consumer}' in topic '{topic}'."
                )

        if experience_count is None and indexes is None:
            raise ValueError("Either experience_count or indexes must be provided for get_experience")

        # 1. Check if multimodal data 'labels' is required.
        if "multimodal" in topic:
            require_mm_labels = "labels" in experience_columns
            experience_columns = [col for col in experience_columns if col != "labels"]
        else:
            require_mm_labels = None

        # 2. Allocate shards and indexes
        shard_map: Dict[str, List[int]] = {}
        if indexes is not None:
            shard_map = ray.get(
                self.manager.allocate_shard_for_indexes.remote(
                    topic,
                    consumer,
                    allow_partial_ready_data,
                    experience_columns,
                    indexes,
                )
            )
        else:
            shard_map = ray.get(
                self.manager.allocate_shard_and_indexes.remote(
                    topic,
                    consumer,
                    allow_partial_ready_data,
                    experience_columns,
                    experience_count,
                    get_n_samples,
                    use_batch_seqlen_balance,
                )
            )
        if not shard_map:
            logger.info(f"Get none experience.")
            return None, None

        # 3. Send GET to each shard and collect results with timeout
        partial_batches: Dict[str, List[Tensor]] = {col: [] for col in experience_columns}
        all_indexes: List[int] = []

        for endpoint, idxs in shard_map.items():
            sock = self._get_socket(endpoint)
            payload = {
                "topic": topic,
                "experience_columns": experience_columns,
                "indexes": idxs
            }
            sock.send_multipart([b"GET", pickle.dumps(payload)])
            reply = sock.recv_multipart()
            if not (reply and reply[0]):
                raise RuntimeError(f"No response received for GET from {endpoint}")
            if reply[0].startswith(b"ERROR:"):
                err = reply[0].decode() if reply else "No reply"
                raise RuntimeError(f"get_experience: TQ_DATA shard error from {endpoint}: {err}")

            response = pickle.loads(reply[0])
            shard_experience: List[List[Tensor]] = deserialize_tensor_lists(response["experience_bytes"])
            returned = response["indexes"]

            # 4. Accumulate per-column lists and global indexes
            for col, lst in zip(experience_columns, shard_experience):
                partial_batches[col].extend(lst)
            all_indexes.extend(returned)

        data_batch = partial_batches

        if len(all_indexes) > 1:
            order = sorted(range(len(all_indexes)), key=all_indexes.__getitem__)
            all_indexes = [all_indexes[i] for i in order]
            for col, lst in data_batch.items():
                data_batch[col] = [lst[i] for i in order]

        # Reshape each column's data using retrieved shapes
        for column, experience in data_batch.items():
            shape_list = ray.get(self.manager.get_shape.remote(topic, column, all_indexes))
            if shape_list:
                data_batch[column] = [exp.reshape(shape) for exp, shape in zip(experience, shape_list)]

        # If multimodal labels are needed, fetch and add to batch
        if require_mm_labels:
            mm_labels = ray.get(self.manager.get_mm_labels.remote(topic, all_indexes))
            data_batch["labels"] = mm_labels
            experience_columns.append("labels")

        # Perform DP padding on data batch and indexes to fit dp_size if required
        if require_dp_padding:
            data_batch, all_indexes = padding_experience_for_dp(data_batch, all_indexes, experience_columns, experience_count, dp_size)
        if pad:
            non_tensor = {}
            tensor_batch = {}
            for key, value in data_batch.items():
                if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                    tensor_batch[key] = value
                else:
                    non_tensor[key] = value
            if tensor_batch:
                data_batch = padding_dict_to_tensor_dict(tensor_batch)
            else:
                data_batch = {}
            data_batch.update(non_tensor)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        ray.get(self.manager.accumulate_timing.remote("get", float(elapsed)))
        return data_batch, all_indexes

    def all_consumed(self, consumer: str, topic: str = DEFAULT_TOPIC, get_n_samples: bool = True) -> bool:
        """Check if the given consumer has consumed all data in the Transfer Queue."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.all_consumed.remote(topic, consumer, get_n_samples))

    def all_updated(self, column: str, topic: str = DEFAULT_TOPIC) -> bool:
        """Check if all data in the specified column has been updated (filled) in the Transfer Queue."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.all_updated.remote(topic, column))

    def increment_ages(self, topic: str = DEFAULT_TOPIC):
        """Increments the age of experiences that have been generated (or partially generated) but not yet consumed."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.increment_ages.remote(topic))

    def reset_consumer_status(self, consumer: str, topic: str = DEFAULT_TOPIC, indexes: List[int] = None):
        """Resets the consumer status for the given consumer and topic for reprocessing."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.reset_consumer_status.remote(consumer, topic, indexes))

    def clear_data_status(self, columns: List[str], indexes: List[int], topic: str = DEFAULT_TOPIC):
        """Clear data readiness status (set to 0) for given columns at specified indexes."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.clear_data_status.remote(topic, indexes, columns))

    def reset_all(self):
        """Fully reset TQ back to post-__init__ state: drop all topics/tables and reset manager state."""
        # NOTE: this removes schemas entirely; add_topic must be called again afterwards.
        ray.get(self.manager.reset_all.remote())
        self.logger.info("TQ_Client: Fully reset TQ to post-__init__ state (ALL topics dropped).")

    def get_columns(self, consumer: str = None, topic: str = DEFAULT_TOPIC) -> List[str]:
        return ray.get(self.manager.get_columns.remote(topic=topic, consumer=consumer))

    def get_metrics(self, topic: str = DEFAULT_TOPIC):
        """Fetch metrics object for a topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_metrics.remote(topic))

    def prefetch_request_index(self, experience_num, topic=DEFAULT_TOPIC):
        """Pre-allocates a contiguous block of request indices for storing processed experience data in the specified topic."""
        return ray.get(self.manager.prefetch_request_index.remote(experience_num, topic))

    def get_cur_index(self, topic: str = DEFAULT_TOPIC):
        return ray.get(self.manager.get_cur_index.remote(topic))

    def update_metrics(self, key: str = "", value=None, cumulate: bool = False, topic: str = DEFAULT_TOPIC):
        """Update metrics for a topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.update_metrics.remote(
            topic, key, value, cumulate=cumulate
        ))

    def create_timing_item(self, name: str) -> None:
        """Create (ensure) a timing item on the manager."""
        ray.get(self.manager.create_timing_item.remote(name))

    def accumulate_timing(self, name: str, seconds: float) -> None:
        """Accumulate elapsed seconds into a timing item on the manager."""
        ray.get(self.manager.accumulate_timing.remote(name, float(seconds)))
    
    def get_timing(self, name: str) -> float:
        """Return a single timing item (seconds)."""
        return ray.get(self.manager.get_timing.remote(name))


    def get_timings(self) -> Dict[str, float]:
        """Return all timing items (a dict of name -> seconds)."""
        return ray.get(self.manager.get_timings.remote())
    
    def reset_timings(self) -> None:
        """Reset all timing items to zero."""
        ray.get(self.manager.reset_timings.remote())

    def get_max_len(self, topic: str = DEFAULT_TOPIC) -> int:
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_max_len.remote(topic))

    def get_partial_rollout_stop_signal(
        self,
        require_max_age_all_finished: bool = True,
        topic: str = DEFAULT_TOPIC
    ) -> bool:
        """Queries the manager to check if partial rollout should be stopped for the given topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_partial_rollout_stop_signal.remote(require_max_age_all_finished, topic))

    def get_incomplete_response_num(self, topic: str = DEFAULT_TOPIC) -> int:
        """Returns the number of incomplete responses in the given topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_incomplete_response_num.remote(topic))

    def put_mm_labels(self, topic: str = DEFAULT_TOPIC, labels: Optional[List[str]] = None, index: List[int] = None):
        """Stores multimodal labels at specified indices in the given topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        if not labels or not index:
            raise ValueError("Both mm_labels and index should be provided.")
        return ray.get(self.manager.put_mm_labels.remote(topic, labels, index))

    def get_mm_labels(self, topic: str = DEFAULT_TOPIC, index: List[int] = None) -> Optional[List[str]]:
        """Retrieves multimodal labels from specified indices in the given topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        if not index:
            return None
        return ray.get(self.manager.get_mm_labels.remote(topic, index))


def get_transfer_queue_client(name: str = "TransferQueueManager") -> TransferQueueClient:
    """
    Get a new TQ_Client instance connected to the named TransferQueueManager actor.
    This uses Ray's global name registry to locate the manager.
    """
    if name is None:
        name = "TransferQueueManager"
    mgr = ray.get_actor(name)
    return TransferQueueClient(mgr)