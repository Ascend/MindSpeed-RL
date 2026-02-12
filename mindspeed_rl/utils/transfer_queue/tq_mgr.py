"""
Transfer Queue Manager for Distributed Experience Coordination

A Ray actor that centrally manages metadata, scheduling, and coordination for a distributed Transfer Queue system. 
Handles topic registration, shard allocation, consumer sampling, and status tracking across TQ_DATA shards.

Maintains per-topic metadata (e.g., readiness, consumption status, age) and delegates data storage to remote TransferQueueShard actors. 
Supports dynamic resizing, batch balancing, multimodal labels, and fine-grained timing metrics with thread-safe, distributed consistency.
"""
import threading
from typing import List, Dict, Optional
import time
from dataclasses import dataclass, field
import torch
from torch import Tensor
import ray

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.metrics import Metric
from mindspeed_rl.utils.transfer_queue.tq_data import TransferQueueShard
from mindspeed_rl.utils.transfer_queue.tq_utils import get_seqlen_balanced_partitions


@dataclass()
class TopicMeta:
    prompts_num: int
    n_samples_per_prompt: int
    nums_tq_data: int
    metrics: Metric
    experience_columns: List[str]
    experience_consumers: List[str]
    timeout: float
    max_age: Optional[int]
    GBS_train: Optional[int]

    max_len: int = field(init=False)
    prompts_per_shard: int = field(init=False)
    prompts_per_shard_last: int = field(init=False)
    per_shard_max_len_list: List[int] = field(init=False)
    shard_sample_offsets: List[int] = field(init=False)

    experience_data_status: Dict[str, Tensor] = field(init=False)
    experience_consumer_status: Dict[str, Tensor] = field(init=False)
    consumer_sampling_lock: Dict[str, threading.Lock] = field(init=False)
    mm_labels: List[str] = field(init=False)
    shape_list: List[torch.Size] = field(init=False)

    def __post_init__(self):
        self.max_len = self.prompts_num * self.n_samples_per_prompt
        if self.nums_tq_data <= 0:
            raise ValueError("nums_tq_data must be > 0")
        base = self.prompts_num // self.nums_tq_data
        extra = self.prompts_num % self.nums_tq_data
        self.prompts_per_shard = base
        self.prompts_per_shard_last = base + extra
        self.seq_len_list = []
        self.enable_partial_rollout = self.max_age > 1 if self.max_age else False
        if self.enable_partial_rollout:
            self.ages = torch.zeros(self.max_len, dtype=torch.int32)
            self.partial_rollout_stop_signal = False

        self.per_shard_max_len_list = [
            (base * self.n_samples_per_prompt) if i < self.nums_tq_data - 1
            else (self.prompts_per_shard_last * self.n_samples_per_prompt)
            for i in range(self.nums_tq_data)
        ]

        self.shard_sample_offsets = []
        running = 0
        for i in range(self.nums_tq_data):
            self.shard_sample_offsets.append(running)
            running += self.per_shard_max_len_list[i]

        self.experience_data_status = {
            col: torch.zeros(self.max_len, dtype=torch.int32) 
            for col in self.experience_columns
        }
        self.consumer_columns = {}
        self.prefetch_request_index_lock = threading.Lock()
        self.cur_index = 0

        self.experience_consumer_status = {
            cons: torch.zeros(self.max_len, dtype=torch.int32) 
            for cons in self.experience_consumers
        }
        self.consumer_sampling_lock = {
            key: threading.Lock() 
            for key in self.experience_consumers
        }
        self.mm_labels = [None] * self.max_len
        self.shape_list = {}


@ray.remote(max_concurrency=100, num_cpus=10, name="TransferQueueManager")
class TransferQueueManager:
    def __init__(self, nums_tq_data: int = 1, base_port: Optional[int] = None) -> None:
        """
        Manager initialization.
        - nums_tq_data: Number of TQ_DATA shards to create.
        - base_port: Starting port number for TQ_DATA servers. If None, shards bind to a random available port.
        """
        self.logger = Loggers(self.__class__.__name__)
        self.nums_tq_data = nums_tq_data

        self.topics: Dict[str, TopicMeta] = {}

        self.data_actors = []        # Ray actor handles for TQ_DATA shards
        self.data_endpoints = []     # ZMQ endpoints (addresses) for each shard

        self.batch_seqlen_balance_mapper = {
            "ref_log_prob": ["prompt_length", "response_length"],
            "actor_log_prob": ["prompt_length", "response_length"],
            "reward_scores": ["prompt_length", "response_length"],
            "actor_train": ["prompt_length", "response_length"]
        }

        all_nodes = [node for node in ray.nodes() if node["Alive"]]
        target_node_ips = [
            node["NodeManagerAddress"]
            for node in all_nodes
            if node["Resources"].get("CPU", 0) >= 1
        ]
        if not target_node_ips:
            raise RuntimeError("No available Ray nodes with CPU resources.")
         
        for i in range(nums_tq_data):
            
            port = None if base_port is None else base_port + i
            
            # Round-robin allocate TQ_DATA
            node_ip = target_node_ips[i % len(target_node_ips)]  
            data_actor = TransferQueueShard.options(
                resources={f"node:{node_ip}": 0.01},
                name=f"TransferQueueShard_{i}"
            ).remote(
                i,
                port,
            )
            self.data_actors.append(data_actor)

        # Fetch endpoints concurrently to avoid serial waits.
        endpoint_refs = [a.get_endpoint.remote() for a in self.data_actors]
        self.data_endpoints = ray.get(endpoint_refs)

        self.logger.info(f"TQ_MGR: Initialized {nums_tq_data} data shards with endpoints: {self.data_endpoints}")

        # Timing accumulators (seconds) stored in a single dict (name -> seconds).
        # Custom items can be added at runtime. Values use 6-decimal precision.
        self._timing_lock = threading.Lock()
        self._timings: Dict[str, float] = {
            "put": 0.0,
            "get": 0.0,
            "dispatch": 0.0,
        }

    def is_running_dapo(self):
        return "sampling" in self.topics

    def register_consumer_columns_dict(
            self,
            columns_dict: Dict[str, List[str]],
            topic) -> None:
        self.topics[topic].consumer_columns = columns_dict

    def get_columns(self, topic, consumer) -> List[str]:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if consumer is None:
            return self.topics[topic].experience_columns
        else:
            return self.topics[topic].consumer_columns[consumer]

    def get_cur_index(self, topic: str):
        return self.topics[topic].cur_index

    def prefetch_request_index(self, experience_num, topic):
        meta_data = self.topics[topic]
        if meta_data.cur_index >= meta_data.max_len:
            return None
        with meta_data.prefetch_request_index_lock:
            request_index = list(range(meta_data.cur_index, min(meta_data.cur_index + experience_num, meta_data.max_len)))
            meta_data.cur_index += experience_num
        return request_index
        
    def add_topic(
        self,
        topic: str,
        prompts_num: int,
        n_samples_per_prompt: int,
        experience_columns: List[str],
        experience_consumers: List[str],
        timeout: float,
        metrics: Metric,
        max_age: int,
        GBS_train: int,
    ) -> None:
        """
        Register a topic and create storage tables on all shards.
        - Pure schema setup; no data is inserted here.
        - Will raise if the topic already exists.
        """
        if topic in self.topics:
            raise ValueError(f"Topic '{topic}' already exists")
        meta = TopicMeta(
            prompts_num=prompts_num,
            n_samples_per_prompt=n_samples_per_prompt,
            nums_tq_data=self.nums_tq_data,
            metrics=metrics,
            experience_columns=experience_columns,
            experience_consumers=experience_consumers,
            timeout=timeout,
            max_age=max_age,
            GBS_train=GBS_train,
        )
        self.topics[topic] = meta
        for i, actor in enumerate(self.data_actors):
            per_shard_max_len_i = meta.per_shard_max_len_list[i]
            global_offset = meta.shard_sample_offsets[i]
            ray.get(actor.add_experience_table.remote(
                topic=topic,
                max_len=per_shard_max_len_i,
                global_offset=global_offset,
                experience_columns=meta.experience_columns,
            ))

    def delete_topic(self, topic: str) -> None:
        """Delete the specified topic by removing its tables from all shards and clearing metadata."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        # Remove tables from each data shard
        for actor in self.data_actors:
            ray.get(actor.remove_experience_table.remote(topic))
        # Remove metadata
        del self.topics[topic]

    def clear_topic(self, topic: str):
        """Clear ONLY the specified topic across all shards and reset its status tracking."""
        # Instruct all TQ_DATA shards to clear this topic
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        # Reset per-topic statuses
        meta = self.topics[topic]
        for actor in self.data_actors:
            ray.get(actor.clear_experience_table.remote(topic=topic))
        meta.experience_data_status = {
            col: torch.zeros(meta.max_len, dtype=torch.int32) 
            for col in meta.experience_columns
        }
        meta.experience_consumer_status = {
            cons: torch.zeros(meta.max_len, dtype=torch.int32) 
            for cons in meta.experience_consumers
        }

        meta.cur_index = 0
        if meta.metrics is not None:
            meta.metrics.reset()

    def refine_topic_for_partial_rollout(self, topic: str):
        """Clears trained experiences and reorders remaining data for the partial rollout"""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]

        trained_indexes = (meta.experience_consumer_status["actor_train"] == 1).nonzero(as_tuple=True)[0]
        if trained_indexes.numel() > 0:
            # Clear trained data, reset status, and reorder them
            all_indexes = torch.arange(meta.max_len)
            non_trained_idx = all_indexes[~torch.isin(all_indexes, trained_indexes)]
            new_order = torch.cat([non_trained_idx, trained_indexes])
            new_order = new_order.tolist()
            meta.ages[trained_indexes] = 0
            meta.ages = meta.ages[new_order]

            for actor in self.data_actors:
                ray.get(actor.clear_experience_table.remote(topic=topic, indexes=trained_indexes))
                ray.get(actor.reorder_experience_table.remote(topic=topic, new_order=new_order))
            for col in meta.experience_columns:
                meta.experience_data_status[col][trained_indexes] = 0
                meta.experience_data_status[col] = meta.experience_data_status[col][new_order]
            for cons in meta.experience_consumer_status:
                meta.experience_consumer_status[cons][trained_indexes] = 0
                meta.experience_consumer_status[cons] = meta.experience_consumer_status[cons][new_order]

        meta.partial_rollout_stop_signal = False
        if meta.metrics is not None:
            meta.metrics.reset()

    def resize_topic(self, topic: str, prompts_num: int, n_samples_per_prompt: int) -> None:
        """
        Resize the specified topic by deleting it and creating a new one with updated sizes.
        Other parameters (columns, consumers, metrics) are preserved.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        # Preserve existing parameters
        old_meta = self.topics[topic]
        experience_columns = old_meta.experience_columns
        experience_consumers = old_meta.experience_consumers
        metrics = old_meta.metrics
        old_meta.cur_index = 0
        # Delete existing topic
        self.delete_topic(topic)
        # Add new topic with updated sizes
        self.add_topic(topic, prompts_num, n_samples_per_prompt,
                       experience_columns, experience_consumers, metrics)
        
    def get_shard_for_indexes(
        self,
        topic: str,
        indexes: List[int]
    ) -> Dict[str, List[int]]:
        """Determine which TQ_DATA shard(s) should handle the given global indexes, and group indexes by shard endpoint."""
        if not indexes:
            raise ValueError("No indexes provided for get_shard_for_indexes.")

        # Validate topic and global range (per-topic)
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if (min(indexes) < 0 and min(indexes) != -2) or max(indexes) >= meta.max_len:
            raise ValueError(f"Indexes out of global range [0, {meta.max_len}).")

        endpoint_map: Dict[str, List[int]] = {ep: [] for ep in self.data_endpoints}

        # Group each index by its shard
        for idx in indexes:
            if idx >= 0:
                shard_id = self._find_shard_id(meta, idx)
                if shard_id is None:
                    raise ValueError(f"Index {idx} does not belong to any shard.")
                endpoint_map[self.data_endpoints[shard_id]].append(idx)

        # Remove shards with no assigned indexes
        return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}


    def allocate_shard_for_indexes(
        self,
        topic: str,
        consumer: str,
        allow_partial_ready_data: bool,
        experience_columns: List[str],
        indexes: List[int],
    ) -> Optional[Dict[str, List[int]]]:
        """Waits for specified indexes to become (partially) ready, marks them as consumed, and returns their shard mapping。"""
        # The criteria for data status to be consumable differ across different consumption modes.
        if allow_partial_ready_data:
            consumable_data_status = [1, 2]
        else:
            consumable_data_status = [1]            
                    
        meta = self.topics[topic]

        if len(indexes) == 0:
            return [[] for _ in range(len(experience_columns))]
        if max(indexes) >= meta.max_len:
            raise ValueError(
                f"Get experience index {max(indexes)} exceeds the Transfer Queue range {meta.max_len}."
            )
        
        start_time = time.time()
        ready_indexes = []
        while True:
            current_ready = []
            for idx in indexes:
                all_ready = True
                for col in experience_columns:
                    if meta.experience_data_status[col][idx] not in consumable_data_status:
                        all_ready = False
                        break
                if all_ready:
                    current_ready.append(idx)

            new_ready = [idx for idx in current_ready if idx not in ready_indexes]
            if new_ready:
                ready_indexes.extend(new_ready)

            if len(ready_indexes) == len(indexes):
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > meta.timeout:
                self.logger.info(f"TIMEOUT: data_ready has slept {elapsed_time} seconds, some indexes may not be ready")
                break

            time.sleep(0.1)

        meta.experience_consumer_status[consumer][ready_indexes] = 1
        return self.get_shard_for_indexes(topic, ready_indexes)

    def allocate_shard_and_indexes(
        self,
        topic: str,
        consumer: str,
        allow_partial_ready_data: bool,
        experience_columns: List[str],
        experience_count: int,
        get_n_samples: bool,
        use_batch_seqlen_balance: bool,
    ) -> Optional[Dict[str, List[int]]]:
        """
        Allocate a set of global indexes for a consumer and group them by shard endpoint.

        Steps:
        1. Validate the consumer and sampling parameters.
        2. Sample ready global indexes (either in multiples of n_samples_per_prompt or freely).
        3. Map each chosen index to its owning shard via shard_sample_offsets.
        4. Group indexes by the shard's ZMQ endpoint.
        5. Return a dict: { endpoint_str: [global_idx, ...], ... }.

        Returns:
            A dict mapping each shard endpoint to the list of global indexes
            that consumer should fetch from that shard. Returns None if no
            indexes are available.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]

        # 1. Validate consumer
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Consumer '{consumer}' not recognized.")

        if experience_count is None:
            raise ValueError("experience_count must be specified when indexes are not provided.")

        if get_n_samples and (experience_count % meta.n_samples_per_prompt != 0):
            raise ValueError(
                f"get_n_samples=True requires experience_count ({experience_count}) "
                f"to be divisible by n_samples_per_prompt ({meta.n_samples_per_prompt})."
            )
        if allow_partial_ready_data and get_n_samples:
            raise ValueError(
                "get_n_samples not supported when allow_partial_ready_data is True"
            )

        # 2. Sample indexes
        if not allow_partial_ready_data:
            if get_n_samples:
                chosen = self._sample_ready_index_n_samples(
                    topic,
                    consumer,
                    experience_count,
                    experience_columns,
                    use_batch_seqlen_balance,
                )
            else:
                chosen = self._sample_ready_index(
                    topic,
                    consumer,
                    experience_count,
                    experience_columns,
                    use_batch_seqlen_balance,
                )
        else:
            if not get_n_samples:
                chosen = self._sample_partial_ready_index(
                    topic,
                    consumer,
                    experience_count,
                    experience_columns,
                )

        if not chosen:
            # No data available
            return None

        # 3. Group by shard
        allocation: Dict[str, List[int]] = {ep: [] for ep in self.data_endpoints}
        for idx in chosen:
            shard_id = self._find_shard_id(meta, idx)
            if shard_id is None:
                raise RuntimeError(f"Index {idx} does not belong to any shard")
            ep = self.data_endpoints[shard_id]
            allocation[ep].append(idx)

        # 4. Prune empty entries
        return {ep: idxs for ep, idxs in allocation.items() if idxs}

    def _sample_ready_index(
            self,
            topic,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
            use_batch_seqlen_balance: bool,
    ) -> Optional[List[int]]:
        """Sample a number of fully ready and unconsumed experience indexes from the specified topic."""
        meta = self.topics[topic]

        with meta.consumer_sampling_lock[consumer]:
            not_consumed_indexes = meta.experience_consumer_status[consumer] == 0
            data_ready_indexes = torch.all(
                torch.stack(
                    [meta.experience_data_status[single_column] == 1 for single_column in experience_columns]
                ), dim=0,
            )

            usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]

            if len(usable_indexes) < experience_count or experience_count <= 0:
                return None

            if consumer in self.batch_seqlen_balance_mapper and use_batch_seqlen_balance and len(
                    usable_indexes) % experience_count == 0:
                sampled_indexes = self.batch_seqlen_balance_sampler(
                    topic, consumer, usable_indexes, experience_count, get_n_samples=False
                )
                if not sampled_indexes:
                    return None
            else:
                sampled_indexes = self.batch_balencing_sampler(
                    topic, experience_columns, usable_indexes, experience_count
                )
            meta.experience_consumer_status[consumer][sampled_indexes] = 1

        return sampled_indexes
    
    def _sample_ready_index_n_samples(
            self,
            topic,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
            use_batch_seqlen_balance: bool,
    ) -> Optional[List[int]]:
        """Samples fully ready experiences grouped by prompt, where each prompt group contains exactly `n_samples_per_prompt` samples."""
        meta = self.topics[topic]
        experience_count_n_samples = experience_count // meta.n_samples_per_prompt
        # update ages
        if meta.enable_partial_rollout:
            self.update_ages_with_group_max(topic, consumer)
        with meta.consumer_sampling_lock[consumer]:
            experience_consumer_status_n_samples = (
                1 - torch.all(
                    torch.reshape(
                        meta.experience_consumer_status[consumer],
                        (meta.prompts_num, meta.n_samples_per_prompt)
                    ) == 0,
                    dim=1
                ).int()
            )
            not_consumed_indexes = experience_consumer_status_n_samples == 0

            experience_data_status_n_samples = {}
            for key, value in meta.experience_data_status.items():
                experience_data_status_n_samples[key] = torch.all(
                    torch.reshape(value, (meta.prompts_num, meta.n_samples_per_prompt)) == 1,
                    dim=1
                ).int()

            data_ready_indexes = torch.all(
                torch.stack([experience_data_status_n_samples.get(col) == 1 for col in experience_columns]),
                dim=0
            )

            usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]
            if len(usable_indexes) < experience_count_n_samples:
                return None

            if meta.enable_partial_rollout:
                if meta.experience_consumer_status[consumer].sum() >= meta.GBS_train * meta.n_samples_per_prompt:
                    return None
                # Sort usable indices by their corresponding age values in descending order
                step = meta.n_samples_per_prompt
                group_represent = [i * step for i in range(meta.prompts_num)]
                group_ages = meta.ages[group_represent]
                # Create a list of tuples (index, age) for usable indexes
                index_age_pairs = [(idx, group_ages[idx]) for idx in usable_indexes]
                # Sort the list of tuples by age in descending order
                sorted_index_age_pairs = sorted(index_age_pairs, key=lambda x: x[1], reverse=True)
                # Extract the sorted indexes
                sorted_usable_indexes = [idx for idx, _ in sorted_index_age_pairs]
                sampled_indexes_n_sample = [int(i) for i in sorted_usable_indexes[:experience_count_n_samples]]
            elif consumer in self.batch_seqlen_balance_mapper and use_batch_seqlen_balance and len(
                    usable_indexes) % experience_count_n_samples == 0:
                sampled_indexes_n_sample = self.batch_seqlen_balance_sampler(
                    topic, consumer, usable_indexes, experience_count_n_samples, get_n_samples=True
                )
                if not sampled_indexes_n_sample:
                    return None
            else:
                sampled_indexes_n_sample = self.batch_balencing_sampler(
                    topic,
                    experience_columns,
                    usable_indexes,
                    experience_count_n_samples,
                )

            sampled_indexes = []
            for n_sample_index in sampled_indexes_n_sample:
                index_list = []
                for index in range(
                        n_sample_index * meta.n_samples_per_prompt,
                        (n_sample_index + 1) * meta.n_samples_per_prompt
                ):
                    index_list.append(index)

                sampled_indexes += index_list
            
            if not sampled_indexes:
                return None
            meta.experience_consumer_status[consumer][sampled_indexes] = 1
            
        return sampled_indexes

    def _sample_partial_ready_index(
            self,
            topic,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
    ) -> Optional[List[int]]:
        """In partial rollout mode, samples experiences that are partially ready and have not yet been consumed."""
        # Data status values 1 or 2 indicating consumable state when partial rollout is enabled
        consumable_data_status = torch.tensor([1, 2])
        
        meta = self.topics[topic]
        self.update_ages_with_group_max(topic, consumer)

        filtered_columns = [
            col
            for col in experience_columns
            if col in meta.experience_data_status
        ]

        with meta.consumer_sampling_lock[consumer]:
            not_consumed_indexes = meta.experience_consumer_status[consumer] == 0
            data_partial_ready_indexes = torch.all(
                torch.stack(
                    [torch.isin(meta.experience_data_status[single_column], consumable_data_status)
                     for single_column in filtered_columns]
                ), dim=0,
            )

            usable_indexes = (not_consumed_indexes & data_partial_ready_indexes).nonzero(as_tuple=True)[0]

            if len(usable_indexes) < experience_count and len(usable_indexes) > 0:
                experience_count = len(usable_indexes)
            if experience_count <= 0:
                return None

            # Sort usable indices by their corresponding age values in descending order
            index_age_pairs = list(zip(usable_indexes, meta.ages[usable_indexes]))
            sorted_index_age_pairs = sorted(index_age_pairs, key=lambda x: x[1], reverse=True)
            sorted_usable_indexes = [idx for idx, _ in sorted_index_age_pairs] 
            sampled_indexes = [int(i) for i in sorted_usable_indexes[:experience_count]]
            meta.experience_consumer_status[consumer][sampled_indexes] = 1

        return sampled_indexes

    def _check_exhaustively_consumed_groups(self, topic: str, consumer: str) -> bool:
        """
        First check that each group's data (each group contains n_prompt replicated entries) is fully consumed,
        then verify that the number of completely consumed groups is greater than GBS.
        """
        repeated_consumer = "actor_rollout"
        multi_consumable_column = "responses"
        if consumer != repeated_consumer:
            raise ValueError(f"Consumer {consumer} is not a registered repeated consumer and cannot access multi-consumable data.")

        meta = self.topics[topic]
        num_groups = meta.max_len // meta.n_samples_per_prompt # self.prompts_num

        # Data being fully ready for the downstream consumer indicates that it has been exhaustively consumed by the upstream consumer
        exhaustively_consumed_status = meta.experience_data_status[multi_consumable_column]
        exhaustively_consumed_groups_mask = (
            exhaustively_consumed_status[:num_groups * meta.n_samples_per_prompt]
            .view(num_groups, meta.n_samples_per_prompt) == 1
        ).all(dim=1)
        exhaustively_consumed_groups_count = exhaustively_consumed_groups_mask.sum().item()

        return exhaustively_consumed_groups_count >= meta.GBS_train

    def all_consumed(self, topic: str, consumer: str, get_n_samples: bool) -> bool:
        """
        Check if the given consumer has consumed all data in the Transfer Queue.
        Returns True if all indices have been consumed by this consumer.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Consumer '{consumer}' not recognized.")

        # Verify if total consumed data volume is sufficient (equal to GBS * n_samples)
        if meta.enable_partial_rollout:
            if meta.GBS_train == 0:
                raise ValueError("GBS for update must be provided when enabling partial rollout")
            if get_n_samples:
                return int(meta.experience_consumer_status[consumer].sum().item()) == meta.GBS_train * meta.n_samples_per_prompt
            else:
                return self._check_exhaustively_consumed_groups(topic, consumer)
        else:
            return int(meta.experience_consumer_status[consumer].sum().item()) == meta.max_len

    def all_updated(self, topic: str, column: str) -> bool:
        """
        Check if all data in the specified column has been updated (filled).
        Returns True if all entries in that column are marked as updated.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if column not in meta.experience_columns:
            raise ValueError(f"Column '{column}' not recognized.")
        return int(meta.experience_data_status[column].sum().item()) == meta.max_len

    def increment_ages(self, topic: str):
        """Increments the age of experiences that have been generated (or partially generated) but not yet consumed."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        meta.ages += (meta.experience_data_status['responses'] != 0).to(torch.int32)

    def reset_consumer_status(self, consumer, topic, indexes):
        """Clears the consumer status at the specified indices for reprocessing, or the entire status array if no indices are specified."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if not indexes:
            meta.experience_consumer_status[consumer] = torch.zeros(meta.max_len, dtype=torch.int32)
        else:
            meta.experience_consumer_status[consumer][indexes] = 0

    def reset_all(self):
        """
        Fully reset to post-__init__ state:
        - Instruct all shards to drop ALL topic tables.
        - Remove ALL topics metadata.
        - Reset timings to initial values.
        """
        # Drop all per-topic tables in every shard (after this, shards have no topics).
        for actor in self.data_actors:
            ray.get(actor.reset_all.remote())
        # Remove manager-side topics and associated runtime state.
        self.topics = {}
        # Reset timings to the same initial keys/values as in __init__.
        with self._timing_lock:
            self._timings = {
                "put": 0.0,
                "get": 0.0,
                "put_prompt": 0.0,
                "dispatch": 0.0,
            }
        self.logger.info("TQ_MGR: Fully reset to post-__init__ state (no topics present).")

    def shutdown(self):
        """Terminate all TQ_DATA actors (optional cleanup)."""
        for actor in self.data_actors:
            ray.kill(actor)
        self.data_actors = []
        self.data_endpoints = []
        self.logger.info("TQ_MGR: All data shards have been shut down.")
    
    def get_shard_sample_counts(self, topic: str) -> List[int]:
        """Return the number of samples assigned to each shard for a topic."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        return list(meta.per_shard_max_len_list)

    def get_all_endpoints(self) -> List[str]:
        """Return the list of all TQ_DATA endpoints (for broadcasting or debugging)."""
        return self.data_endpoints

    def get_n_samples_per_prompt(self, topic: str) -> int:
        """Return _n_samples_per_prompt for the given topic."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        return self.topics[topic].n_samples_per_prompt
    
    def update_data_status(self, topic: str, indexes: List[int], columns: List[str], data_status: str = "ready") -> None:
        """
        Update the data readiness status for given columns at specified indexes.
        This should be called by TQ_DATA shards after they finish storing new data.
        """
        if not indexes or not columns:
            return
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        # Mark each specified column at those index positions
        for col in columns:
            if col in meta.experience_data_status:
                if data_status == "ready":
                    # Set status to 1 (ready) for all given indexes in this column
                    meta.experience_data_status[col][indexes] = 1  # advanced indexing on torch tensor
                if data_status == "partial_ready":
                    # Set status to 2 (partial-ready) for all given indexes in this column
                    meta.experience_data_status[col][indexes] = 2
            else:
                raise ValueError(f"Column '{col}' not recognized.")

    def clear_data_status(self, topic: str, indexes: List[int], columns: List[str]) -> None:
        """Clear data readiness status (set to 0) for given columns at specified indexes."""
        if not indexes or not columns:
            return
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        for col in columns:
            if col in meta.experience_data_status:
                meta.experience_data_status[col][indexes] = 0
            else:
                raise ValueError(f"Column '{col}' not recognized.")
    
    def update_ages_with_group_max(self, topic: str, consumer: str = "actor_train"):
        """Sets the age of all samples in a prompt group to the maximum age within that group."""
        meta = self.topics[topic]
        with meta.consumer_sampling_lock[consumer]:
            group_indices = torch.arange(0, meta.max_len, meta.n_samples_per_prompt)  # e.g: should be [0, 8, 16] when n=8         
            group_ages = []
            for i in group_indices:
                group_ages.append(meta.ages[i:i + meta.n_samples_per_prompt].max())
                meta.ages[i:i + meta.n_samples_per_prompt] = group_ages[-1]

    def get_metrics(self, topic: str):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        return self.topics[topic].metrics

    def update_metrics(self, topic: str, key="", value=None, cumulate=False):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        metrics = self.topics[topic].metrics
        if metrics is None:
            return
        metrics.update(key, value, cumulate=cumulate)

    def init_ready(self):
        return True
    
    def create_timing_item(self, name: str) -> None:
        """Create (or ensure) a custom timing item initialized to 0.0 seconds."""
        if not isinstance(name, str) or not name:
            raise ValueError("Timing item name must be a non-empty string")
        with self._timing_lock:
            if name in self._timings:
                raise ValueError(f"Timing item '{name}' already exists")
            self._timings[name] = 0.0
    
    def accumulate_timing(self, name: str, seconds: float) -> None:
        """
        Add elapsed time (in seconds) to a timing item.
        Value should come from time.perf_counter() deltas. Stored with 6-decimal precision.
        """
        val = round(float(seconds), 6)
        with self._timing_lock:
            cur = self._timings.get(name, 0.0)
            self._timings[name] = round(cur + val, 6)

    def get_timing(self, name: str) -> float:
        """Get accumulated time in seconds (6-decimal precision) for the given name."""
        with self._timing_lock:
            if name in self._timings:
                return round(self._timings[name], 6)
        raise ValueError(f"Unknown timing name '{name}'")
    
    def get_timings(self) -> Dict[str, float]:
        """
        Return a dict of all accumulated timings (seconds, 6-decimal precision),
        including built-ins and any custom items.
        """
        with self._timing_lock:
            return {k: round(v, 6) for k, v in self._timings.items()}
        
    def reset_timings(self) -> None:
        """Reset all accumulated timings to zero."""
        with self._timing_lock:
            self._timings = {k: 0.0 for k in self._timings.keys()}

    def _find_shard_id(self, meta: TopicMeta, global_index: int) -> Optional[int]:
        if global_index < 0 or global_index >= meta.max_len:
            return None
        for i in range(self.nums_tq_data):
            lo = meta.shard_sample_offsets[i]
            hi = lo + meta.per_shard_max_len_list[i]
            if lo <= global_index < hi:
                return i
        return None

    def get_max_len(self, topic: str) -> int:
        meta = self.topics[topic]
        return meta.max_len

    def _fetch_column_values(self, topic: str, column: str, global_indexes: List[int]):
        """Fetch column values for arbitrary global indexes across shards via Ray."""
        if not global_indexes:
            return []
        meta = self.topics[topic]
        shard_groups = {}  # sid -> list[(pos, gi)]
        for pos, gi in enumerate(global_indexes):
            sid = self._find_shard_id(meta, gi)
            if sid is None:
                raise RuntimeError(f"Global index {gi} does not belong to any shard.")
            if sid not in shard_groups:
                shard_groups[sid] = []
            shard_groups[sid].append((pos, gi))

        # fire RPCs
        pending = {}
        for sid, pairs in shard_groups.items():
            idxs = [gi for _, gi in pairs]
            pending[sid] = self.data_actors[sid].get_values.remote(topic, column, idxs)

        per_shard_results = {sid: ray.get(obj) for sid, obj in pending.items()}

        # stitch back to original order
        out = [None] * len(global_indexes)
        for sid, pairs in shard_groups.items():
            vals = per_shard_results[sid]
            for local_i, (pos, _) in enumerate(pairs):
                out[pos] = vals[local_i]
        return out

    def batch_seqlen_balance_sampler(
        self, topic, consumer, usable_indexes, experience_count, get_n_samples=False
    ):
        """
        Same behavior as legacy sampler, except values come from RPC.
        We first prefetch all needed values into a local experience_data dict:
            experience_data[col][global_index] -> Tensor
        Then reuse the legacy summation logic.
        """
        meta = self.topics[topic]

        if len(usable_indexes) == experience_count:
            sampled_indexes = [int(usable_indexes[i].item()) for i in range(experience_count)]
            return sampled_indexes

        seq_len_columns = self.batch_seqlen_balance_mapper.get(consumer)

        # 1) Collect all global indexes we will need
        needed_indexes = []
        if get_n_samples:
            # prompt-level indexes; expand to sample-level
            nsp = meta.n_samples_per_prompt
            for idx in usable_indexes:
                base = int(idx.item()) * nsp
                for addition in range(nsp):
                    needed_indexes.append(base + addition)
        else:
            # sample-level indexes
            for idx in usable_indexes:
                needed_indexes.append(int(idx.item()))

        # 2) Prefetch via RPC into local experience_data
        experience_data = {}
        for key in seq_len_columns:
            experience_data[key] = {}
            # fetch in the same order as needed_indexes to preserve alignment
            vals = self._fetch_column_values(topic, key, needed_indexes)
            # stitch into a dict keyed by global index
            for i, gi in enumerate(needed_indexes):
                experience_data[key][gi] = vals[i]

        # 3) Build seq_len_list
        if get_n_samples:
            seq_len_list = []
            nsp = meta.n_samples_per_prompt
            for idx in usable_indexes:
                idx_val = int(idx.item())
                total_len = 0
                for addition in range(nsp):
                    for key in seq_len_columns:
                        total_len += experience_data.get(key, {}).get(idx_val * nsp + addition, 0).item()
                seq_len_list.append(total_len)
        else:
            seq_len_list = []
            for idx in usable_indexes:
                idx_val = int(idx.item())
                total_len = 0
                for key in seq_len_columns:
                    total_len += experience_data.get(key, {}).get(idx_val, 0).item()
                seq_len_list.append(total_len)

        k_partitions = len(seq_len_list) // experience_count
        sampled_indexes_idx = get_seqlen_balanced_partitions(seq_len_list, k_partitions, equal_size=True)
        if len(sampled_indexes_idx) > 0:
            sampled_indexes = [int(usable_indexes[i].item()) for i in sampled_indexes_idx[0]]
        else:
            sampled_indexes = None
        return sampled_indexes


    def batch_balencing_sampler(
        self, topic, experience_columns, usable_indexes, experience_count, target_seq_len=None
    ):
        """
        Performs weighted sampling favoring indexes whose associated samples have a total sequence length close to `target_seq_len`; 
        falls back to uniform sampling when `target_seq_len` is unspecified.
        """
        if target_seq_len is None:
            weights = torch.ones(len(usable_indexes))
        else:
            meta = self.topics[topic]

            # 1) Collect all needed global indexes (sample-level)
            needed_indexes = [int(idx.item()) for idx in usable_indexes]

            # 2) Prefetch via RPC into a local experience_data dict:
            #    experience_data[col][gi] -> Tensor
            experience_data = {}
            for key in experience_columns:
                experience_data[key] = {}
                vals = self._fetch_column_values(topic, key, needed_indexes)
                for i, gi in enumerate(needed_indexes):
                    experience_data[key][gi] = vals[i]


            seq_len = torch.tensor(
                [
                    sum([experience_data[key][gi].numel() for key in experience_columns])
                    for gi in needed_indexes
                ]
            )
            weights = torch.sigmoid(1 / (torch.abs(seq_len - target_seq_len) + 0.001))

        sampled_indexes_idx = torch.multinomial(weights, experience_count, replacement=False).tolist()
        sampled_indexes = [int(usable_indexes[i].item()) for i in sampled_indexes_idx]
        
        return sampled_indexes

    def get_partial_rollout_stop_signal(self, require_max_age_all_finished, topic):
        """
        Determines whether to stop the partial rollout by checking if enough prompt groups have all their responses fully generated, 
        and—if required—that all responses at the maximum allowed age have also been fully generated.
        """
        consumer = "actor_rollout"
        partial_rollout_column = "responses"

        meta = self.topics[topic]
        rollout_exhaustively_consumed = self._check_exhaustively_consumed_groups(topic, consumer)

        if require_max_age_all_finished:
            max_age_index = (meta.ages == meta.max_age - 1).nonzero(as_tuple=True)[0]
            max_age_all_finished = (meta.experience_data_status[partial_rollout_column][max_age_index] == 1).sum().item() == len(max_age_index)
            meta.partial_rollout_stop_signal = rollout_exhaustively_consumed & max_age_all_finished
        else:
            meta.partial_rollout_stop_signal = rollout_exhaustively_consumed
        return meta.partial_rollout_stop_signal

    def get_incomplete_response_num(self, topic):
        """Returns the number of prompts whose responses have been partially generated but are not yet complete."""
        meta = self.topics[topic]
        incomplete_response_num = (meta.experience_data_status['prompts'] != 0).sum().item() - (meta.experience_data_status['responses'] == 1).sum().item()

        return incomplete_response_num

    def put_mm_labels(self, topic, labels, index):
        """Stores multimodal labels at specified global indexes for the given topic."""
        meta = self.topics[topic]
        if len(labels) != len(index):
            raise ValueError(f"The length of labels({len(labels)}) doesn't match the length of index({len(index)}).")
        for i, idx in enumerate(index):
            meta.mm_labels[idx] = labels[i]

    def get_mm_labels(self, topic, index):
        """Retrieves multimodal labels stored at the specified global indexes for the given topic."""
        meta = self.topics[topic]
        return [meta.mm_labels[idx] for idx in index]

    def store_shape(self, topic, column, indexes, shape_list):
        """Stores the shapes of tensors in an experience column at specified indexes to enable shape reconstruction upon retrieval."""
        if len(shape_list) != len(indexes):
            raise ValueError("The length of indexes is not equal to the length of shape_list.")
        meta = self.topics[topic]
        if column not in meta.shape_list.keys():
            meta.shape_list[column] = [None] * meta.max_len
        for idx, shape in zip(indexes, shape_list):
            meta.shape_list[column][idx] = shape

    def get_shape(self, topic, column, indexes):
        """Retrieves stored tensor shapes for an experience column at the specified indexes."""
        meta = self.topics[topic]
        if column not in meta.shape_list.keys():
            return []
        return [meta.shape_list[column][idx] for idx in indexes]

