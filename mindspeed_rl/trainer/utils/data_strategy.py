# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import List, Optional

import ray
import torch

from mindspeed_rl.trainer.utils.transfer_dock import GRPOTransferDock
from mindspeed_rl.trainer.utils.mm_transfer_dock import MMGRPOTransferDock, unpack_mm_experience as unpack_mm_experience_td
from mindspeed_rl.utils.transfer_queue.tq_mgr import TransferQueueManager
from mindspeed_rl.utils.transfer_queue.tq_client import get_transfer_queue_client
from mindspeed_rl.utils.transfer_queue.tq_utils import (
    prepare_batch_mm,
    prepare_dummy_response,
    unpack_mm_experience as unpack_mm_experience_tq,
)
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.utils import is_multimodal

logger = Loggers("data_strategy")


_STRATEGY_TD = "td"
_STRATEGY_TQ = "tq"

_STRATEGY_ALIASES = {
    "transfer_dock": _STRATEGY_TD,
    "dock": _STRATEGY_TD,
    "td": _STRATEGY_TD,
    "transfer_queue": _STRATEGY_TQ,
    "queue": _STRATEGY_TQ,
    "tq": _STRATEGY_TQ,
}


def unpack_mm_experience(batch_data):
    """Unified MM unpacker for TD/TQ."""
    if not batch_data:
        return {}
    try:
        return unpack_mm_experience_tq(batch_data)
    except Exception:
        return unpack_mm_experience_td(batch_data)


def _normalize_strategy(value: Optional[str]) -> str:
    if not value:
        return _STRATEGY_TD
    normalized = _STRATEGY_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported data_strategy: {value}")
    return normalized


@dataclass
class TransferDocks:
    td: object = None
    mm_td: object = None
    sampling_td: object = None
    mm_sampling_td: object = None


class _RemoteCallable:
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def remote(self, *args, **kwargs):
        return ray.put(self._func(*args, **kwargs))


class SyncTransferQueueAdapter:
    """Sync wrapper to adapt TransferQueue to TransferDock-like API."""

    def __init__(
        self,
        topic: str,
        enable_partial_rollout: bool = False,
        strict_partial_rollout: bool = False,
    ) -> None:
        self.topic = topic
        self.enable_partial_rollout = enable_partial_rollout
        # Align TQ partial rollout behavior with TD when enabled (GRPO only).
        self.strict_partial_rollout = strict_partial_rollout
        # Lazily create TransferQueueClient in the target process.
        # This avoids pickling live ZMQ handles across processes.
        self.tq = None
        self._init_remote_callables()

    def _init_remote_callables(self):
        self.get_columns = _RemoteCallable(self._get_columns)
        self.get_experience = _RemoteCallable(self._get_experience)
        self.get_experience_dict = _RemoteCallable(self._get_experience_dict)
        self.put_experience = _RemoteCallable(self._put_experience)
        self.clear = _RemoteCallable(self._clear)
        self.all_consumed = _RemoteCallable(self._all_consumed)
        self.get_metrics = _RemoteCallable(self._get_metrics)
        self.update_metrics = _RemoteCallable(self._update_metrics)
        self.prefetch_request_index = _RemoteCallable(self._prefetch_request_index)
        self.get_cur_index = _RemoteCallable(self._get_cur_index)
        self.get_update_ready = _RemoteCallable(self._get_update_ready)
        self.get_incomplete_response_num = _RemoteCallable(self._get_incomplete_response_num)

    def _ensure_tq(self):
        if self.tq is None:
            self.tq = get_transfer_queue_client()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tq"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_remote_callables()

    def is_transfer_queue(self) -> bool:
        return True

    def _get_columns(self, consumer: str = None):
        self._ensure_tq()
        return self.tq.get_columns(consumer=consumer, topic=self.topic)

    def _get_experience(
        self,
        consumer: str,
        experience_columns: List[str],
        experience_count: int = None,
        dp_size: int = 1,
        indexes: List[int] = None,
        get_n_samples: bool = True,
        use_batch_seqlen_balance: bool = False,
        allow_partial_ready_data: bool = False,
    ):
        self._ensure_tq()
        require_dp_padding = dp_size > 1 and allow_partial_ready_data
        batch, ret_indexes = self.tq.get_experience(
            consumer=consumer,
            experience_columns=experience_columns,
            experience_count=experience_count,
            indexes=indexes,
            get_n_samples=get_n_samples,
            topic=self.topic,
            use_batch_seqlen_balance=use_batch_seqlen_balance,
            dp_size=dp_size,
            require_dp_padding=require_dp_padding,
            allow_partial_ready_data=allow_partial_ready_data,
        )
        is_strict_partial_rollout = self.strict_partial_rollout and self.enable_partial_rollout
        if is_strict_partial_rollout:
            should_clear_responses = consumer == "actor_rollout" and ret_indexes
        else:
            should_clear_responses = False
        if should_clear_responses:
            try:
                valid_indexes = [idx for idx in ret_indexes if isinstance(idx, int) and idx >= 0]
            except Exception:
                valid_indexes = []
            if valid_indexes:
                self.tq.clear_data_status(
                    columns=["responses", "response_length"],
                    indexes=valid_indexes,
                    topic=self.topic,
                )
        return batch, ret_indexes

    def _get_experience_dict(
        self,
        experience_columns: List[str],
        indexes: List[int] = None,
        get_n_samples: bool = True,
    ):
        self._ensure_tq()
        batch, _ = self.tq.get_experience(
            consumer="dynamic_sampling",
            experience_columns=experience_columns,
            indexes=indexes,
            get_n_samples=get_n_samples,
            topic=self.topic,
            pad=False,
        )
        return batch or {}

    def _put_experience(
        self,
        data_dict,
        indexes: List[int] = None,
        is_prompt: bool = False,
        unpad: bool = True,
        data_status: str = "ready",
    ):
        self._ensure_tq()
        rollout_completed = None
        if "rollout_completed" in data_dict:
            # TQ does not store rollout_completed; TD uses it only for status updates.
            data_dict = dict(data_dict)
            rollout_completed = data_dict.pop("rollout_completed", None)
        self.tq.put_experience(
            data_dict=data_dict,
            indexes=indexes,
            unpad=unpad,
            topic=self.topic,
            data_status=data_status,
        )
        if is_prompt and self.enable_partial_rollout:
            dummy_response = prepare_dummy_response(indexes=indexes)
            dummy_status = "ready" if self.strict_partial_rollout else "partial_ready"
            self.tq.put_experience(
                data_dict=dummy_response,
                indexes=indexes,
                topic=self.topic,
                data_status=dummy_status,
            )
        is_strict_partial_rollout = self.strict_partial_rollout and self.enable_partial_rollout
        if is_strict_partial_rollout:
            should_reset_consumer = rollout_completed is not None and "responses" in data_dict and indexes
        else:
            should_reset_consumer = False
        if should_reset_consumer:
            # Match TD behavior: allow incomplete samples to be re-consumed.
            if torch.is_tensor(rollout_completed):
                rollout_list = rollout_completed.detach().cpu().tolist()
            else:
                rollout_list = rollout_completed
            incomplete_indexes = []
            for idx, done in zip(indexes, rollout_list):
                value = done[0] if isinstance(done, (list, tuple)) else done
                if int(value) == 0:
                    incomplete_indexes.append(idx)
            if incomplete_indexes:
                self.tq.reset_consumer_status(
                    consumer="actor_rollout", topic=self.topic, indexes=incomplete_indexes
                )

    def _clear(self, consumer: str = "actor_train"):
        self._ensure_tq()
        if consumer in (None, "actor_train"):
            if self.enable_partial_rollout:
                self.tq.refine_topic_for_partial_rollout(topic=self.topic)
                self.tq.increment_ages(topic=self.topic)
            else:
                self.tq.clear_topic(topic=self.topic)
        else:
            self.tq.reset_consumer_status(consumer=consumer, topic=self.topic)

    def _all_consumed(self, consumer: str, get_n_samples: bool = True):
        self._ensure_tq()
        return self.tq.all_consumed(consumer=consumer, topic=self.topic, get_n_samples=get_n_samples)

    def _get_metrics(self):
        self._ensure_tq()
        return self.tq.get_metrics(topic=self.topic)

    def _update_metrics(self, key: str = "", value=None, cumulate: bool = False):
        self._ensure_tq()
        self.tq.update_metrics(topic=self.topic, key=key, value=value, cumulate=cumulate)

    def _prefetch_request_index(self, experience_num: int):
        self._ensure_tq()
        return self.tq.prefetch_request_index(experience_num=experience_num, topic=self.topic)

    def _get_cur_index(self):
        self._ensure_tq()
        return self.tq.get_cur_index(topic=self.topic)

    def _get_update_ready(self, require_max_age_all_finished: bool = True):
        self._ensure_tq()
        return self.tq.get_partial_rollout_stop_signal(
            require_max_age_all_finished=require_max_age_all_finished,
            topic=self.topic,
        )

    def _get_incomplete_response_num(self):
        self._ensure_tq()
        return self.tq.get_incomplete_response_num(topic=self.topic)


class DataStrategy:
    """
    Factory and wrapper for transfer backends.

    For now, TQ and TD share the same underlying dock implementation. The strategy
    flag keeps the construction logic centralized for future expansion.
    """

    def __init__(self, rl_config, strategy: Optional[str] = None):
        self.rl_config = rl_config
        self.strategy = _normalize_strategy(strategy or getattr(rl_config, "data_strategy", None))
        logger.info(f"DataStrategy initialized with strategy: {self.strategy}")
        self.docks = TransferDocks()
        self.tq_mgr = None
        self.tq = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop TransferQueueClient (holds ZMQ/C++ handles) before cross-process transfer.
        state["tq"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_tq_manager(self):
        if self.tq_mgr is not None:
            return
        try:
            self.tq_mgr = ray.get_actor("TransferQueueManager")
        except ValueError:
            self.tq_mgr = TransferQueueManager.remote(
                self.rl_config.transfer_queue_data_shard_num,
                self.rl_config.transfer_queue_data_shard_port_base,
            )
            ray.get(self.tq_mgr.init_ready.remote())
        self.tq = get_transfer_queue_client()

    @property
    def td(self):
        return self.docks.td

    @property
    def mm_td(self):
        return self.docks.mm_td

    @property
    def sampling_td(self):
        return self.docks.sampling_td

    @property
    def mm_sampling_td(self):
        return self.docks.mm_sampling_td

    def build_grpo(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        metrics,
        max_age: int,
        gbs_train: int,
        addition_columns: Optional[List[str]] = None,
        reuse_image_embeds: bool = False,
        dataset_additional_keys: Optional[List[str]] = None,
    ) -> TransferDocks:
        if self.strategy == _STRATEGY_TQ:
            self._ensure_tq_manager()
            experience_columns = [
                "prompts",
                "prompt_length",
                "responses",
                "response_length",
                "attention_mask",
                "labels",
                "input_ids",
                "input_ids_length",
                "actor_rollout",
                "rm_scores",
                "token_level_rewards",
                "old_log_prob",
                "ref_log_prob",
                "advantages",
                "returns",
            ]
            experience_consumers = [
                "trainer",
                "actor_rollout",
                "actor_image_embeds",
                "actor_log_prob",
                "ref_log_prob",
                "actor_train",
                "compute_advantage",
                "rule_reward",
                "reward_scores",
                "grpo_metrics",
            ]
            if addition_columns:
                for column in addition_columns:
                    if column not in experience_columns:
                        experience_columns.append(column)
            if self.rl_config.multi_turn_enable:
                experience_columns.extend(["response_mask", "tool_call_num"])
            self.tq.add_topic(
                prompts_num=prompts_num,
                n_samples_per_prompt=n_samples_per_prompt,
                experience_columns=experience_columns,
                experience_consumers=experience_consumers,
                metrics=metrics,
                max_age=max_age,
                GBS_train=gbs_train,
            )
            additional_keys = dataset_additional_keys or []
            consumer_columns = {
                "actor_rollout": ["prompts", "prompt_length"],
                "actor_image_embeds": ["input_ids"],
                "actor_log_prob": ["input_ids", "responses", "response_length", "prompt_length"],
                "ref_log_prob": ["input_ids", "responses", "response_length", "prompt_length"],
                "actor_train": ["responses", "advantages", "old_log_prob", "ref_log_prob", "input_ids", "response_length", "prompt_length"],
                "rule_reward": ["prompts", "responses", "response_length", *additional_keys],
                "reward_scores": ["input_ids", "prompt_length", "responses", "response_length", *additional_keys],
                "grpo_metrics": ["rm_scores", "responses", "advantages", "returns", "prompt_length", "response_length"],
            }
            if self.rl_config.adv_estimator == "gae":
                consumer_columns["compute_advantage"] = ["values", "responses", "token_level_rewards", "response_length"]
                if not self.rl_config.use_kl_in_reward:
                    consumer_columns["compute_advantage"] = ["values", "responses", "rm_scores", "response_length"]
            else:
                consumer_columns["compute_advantage"] = ["responses", "rm_scores", "response_length"]
            if is_multimodal():
                consumer_columns["actor_rollout"].extend(["input_ids", "input_ids_length"])
                consumer_columns["actor_log_prob"].extend(["attention_mask", "position_ids", "input_ids_length"])
                consumer_columns["ref_log_prob"].extend(["attention_mask", "position_ids", "input_ids_length"])
                consumer_columns["actor_train"].extend(["attention_mask", "position_ids"])
                mm_experience_columns = [
                    "image",
                    "pixel_values",
                    "image_grid_thw",
                    "image_shape",
                    "labels",
                    "vit_embeds",
                    "position_ids",
                    "image_num",
                    "video",
                    "video_shape",
                    "video_fps",
                    "video_num",
                ]
                self.tq.add_topic(
                    prompts_num=prompts_num,
                    n_samples_per_prompt=n_samples_per_prompt,
                    experience_columns=mm_experience_columns,
                    experience_consumers=experience_consumers,
                    metrics=metrics,
                    topic="multimodal",
                )
                mm_consumer_columns = {
                    "actor_rollout": ["image", "image_shape", "image_num", "video", "video_shape", "video_fps", "video_num"],
                    "actor_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
                    "ref_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
                    "actor_train": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
                    "rule_reward": ["labels"],
                }
                if reuse_image_embeds:
                    mm_consumer_columns["actor_image_embeds"] = ["pixel_values", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["actor_rollout"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["actor_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["ref_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["actor_train"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                self.tq.register_consumer_columns_dict(mm_consumer_columns, "multimodal")
            if max_age > 1:
                consumer_columns["actor_rollout"].extend(["responses", "response_length"])
            self.tq.register_consumer_columns_dict(consumer_columns)
            self.docks.td = SyncTransferQueueAdapter(
                topic="default_topic",
                enable_partial_rollout=max_age > 1,
                strict_partial_rollout=True,
            )
            if is_multimodal():
                self.docks.mm_td = SyncTransferQueueAdapter(
                    topic="multimodal",
                    enable_partial_rollout=max_age > 1,
                    strict_partial_rollout=True,
                )
        else:
            self.docks.td = GRPOTransferDock.remote(
                prompts_num=prompts_num,
                n_samples_per_prompt=n_samples_per_prompt,
                metrics=metrics,
                max_age=max_age,
                GBS_train=gbs_train,
                addition_columns=addition_columns,
            )
            if is_multimodal():
                self.docks.mm_td = MMGRPOTransferDock.remote(
                    prompts_num,
                    n_samples_per_prompt,
                    reuse_image_embeds,
                )
        return self.docks

    def build_ppo(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        metrics,
        addition_columns: Optional[List[str]] = None,
        addition_consumers: Optional[List[str]] = None,
        dataset_additional_keys: Optional[List[str]] = None,
    ) -> TransferDocks:
        if self.strategy == _STRATEGY_TQ:
            self._ensure_tq_manager()
            experience_columns = [
                "prompts",
                "prompt_length",
                "responses",
                "response_length",
                "attention_mask",
                "labels",
                "input_ids",
                "input_ids_length",
                "actor_rollout",
                "rm_scores",
                "token_level_rewards",
                "old_log_prob",
                "ref_log_prob",
                "advantages",
                "returns",
            ]
            experience_consumers = [
                "trainer",
                "actor_rollout",
                "actor_log_prob",
                "ref_log_prob",
                "actor_train",
                "compute_advantage",
                "rule_reward",
                "reward_scores",
                "ppo_metrics",
            ]
            if addition_columns:
                for column in addition_columns:
                    if column not in experience_columns:
                        experience_columns.append(column)
            if addition_consumers:
                for consumer in addition_consumers:
                    if consumer not in experience_consumers:
                        experience_consumers.append(consumer)
            if self.rl_config.multi_turn_enable:
                experience_columns.extend(["response_mask", "tool_call_num"])
            self.tq.add_topic(
                prompts_num=prompts_num,
                n_samples_per_prompt=n_samples_per_prompt,
                experience_columns=experience_columns,
                experience_consumers=experience_consumers,
                metrics=metrics,
                timeout=300.0,
            )
            additional_keys = dataset_additional_keys or []
            consumer_columns = {
                "actor_rollout": ["prompts", "prompt_length"],
                "actor_log_prob": ["input_ids", "responses", "response_length", "prompt_length"],
                "ref_log_prob": ["input_ids", "responses", "response_length", "prompt_length"],
                "actor_train": ["responses", "advantages", "old_log_prob", "ref_log_prob", "input_ids", "response_length", "prompt_length"],
                "rule_reward": ["prompts", "responses", "response_length", *additional_keys],
                "reward_scores": ["input_ids", "prompt_length", "responses", "response_length", *additional_keys],
                "ppo_metrics": ["rm_scores", "responses", "advantages", "returns", "prompt_length", "response_length"],
            }
            if self.rl_config.adv_estimator == "gae":
                consumer_columns["compute_advantage"] = ["values", "responses", "token_level_rewards", "response_length"]
                if not self.rl_config.use_kl_in_reward:
                    consumer_columns["compute_advantage"] = ["values", "responses", "rm_scores", "response_length"]
            else:
                consumer_columns["compute_advantage"] = ["responses", "rm_scores", "response_length"]
            self.tq.register_consumer_columns_dict(consumer_columns)
            self.docks.td = SyncTransferQueueAdapter(topic="default_topic")
        else:
            self.docks.td = GRPOTransferDock.remote(
                prompts_num,
                n_samples_per_prompt,
                metrics,
                addition_columns=addition_columns,
                addition_consumers=addition_consumers,
            )
        return self.docks

    def build_dapo(
        self,
        should_filter: bool,
        max_num_prompt_in_batch: int,
        td_max_len: int,
        n_samples_per_prompt: int,
        metrics,
        max_age: int,
        gbs_train: int,
        addition_columns: Optional[List[str]] = None,
        addition_consumers: Optional[List[str]] = None,
        dataset_additional_keys: Optional[List[str]] = None,
    ) -> TransferDocks:
        if self.strategy == _STRATEGY_TQ:
            self._ensure_tq_manager()
            experience_columns = [
                "prompts",
                "prompt_length",
                "responses",
                "response_length",
                "attention_mask",
                "labels",
                "input_ids",
                "input_ids_length",
                "actor_rollout",
                "rm_scores",
                "token_level_rewards",
                "old_log_prob",
                "ref_log_prob",
                "advantages",
                "returns",
            ]
            experience_consumers = [
                "trainer",
                "actor_rollout",
                "actor_image_embeds",
                "actor_log_prob",
                "ref_log_prob",
                "actor_train",
                "compute_advantage",
                "rule_reward",
                "reward_scores",
            ]
            if addition_columns:
                for column in addition_columns:
                    if column not in experience_columns:
                        experience_columns.append(column)
            if addition_consumers:
                for consumer in addition_consumers:
                    if consumer not in experience_consumers:
                        experience_consumers.append(consumer)
            additional_keys = dataset_additional_keys or []
            consumer_columns = {
                "actor_rollout": ["prompts", "prompt_length"],
                "actor_image_embeds": ["input_ids"],
                "actor_log_prob": ["input_ids", "responses", "response_length", "prompt_length"],
                "ref_log_prob": ["input_ids", "responses", "response_length", "prompt_length"],
                "actor_train": ["responses", "advantages", "old_log_prob", "ref_log_prob", "input_ids", "response_length", "prompt_length"],
                "rule_reward": ["prompts", "responses", "response_length", *additional_keys],
                "reward_scores": ["input_ids", "prompt_length", "responses", "response_length", *additional_keys],
                "dapo_metrics": ["rm_scores", "responses", "advantages", "returns", "prompt_length", "response_length"],
            }
            if max_age > 1:
                consumer_columns["actor_rollout"].extend(["responses", "response_length", "age"])
            if is_multimodal():
                consumer_columns["actor_log_prob"].extend(["attention_mask", "position_ids", "input_ids_length"])
                consumer_columns["ref_log_prob"].extend(["attention_mask", "position_ids", "input_ids_length"])
                consumer_columns["actor_train"].extend(["attention_mask", "position_ids"])
                mm_experience_columns = [
                    "image",
                    "pixel_values",
                    "image_grid_thw",
                    "image_shape",
                    "labels",
                    "vit_embeds",
                    "position_ids",
                    "image_num",
                    "video",
                    "video_shape",
                    "video_fps",
                    "video_num",
                ]
                mm_consumer_columns = {
                    "actor_rollout": ["image", "image_shape", "image_num", "video", "video_shape", "video_fps", "video_num"],
                    "actor_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
                    "ref_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
                    "actor_train": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
                    "rule_reward": ["labels"],
                }
                if self.rl_config.reuse_image_embeds:
                    mm_consumer_columns["actor_image_embeds"] = ["pixel_values", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["actor_rollout"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["actor_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["ref_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
                    mm_consumer_columns["actor_train"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
            if self.rl_config.multi_turn_enable:
                experience_columns.extend(["response_mask", "tool_call_num"])
            if should_filter:
                self.tq.add_topic(
                    max_num_prompt_in_batch,
                    n_samples_per_prompt,
                    metrics=metrics,
                    experience_columns=experience_columns,
                    timeout=300.0,
                    experience_consumers=experience_consumers,
                )
                self.tq.add_topic(
                    td_max_len,
                    n_samples_per_prompt,
                    metrics=metrics,
                    experience_columns=experience_columns,
                    timeout=300.0,
                    experience_consumers=experience_consumers,
                    topic="sampling",
                )
                if is_multimodal():
                    self.tq.add_topic(
                        prompts_num=td_max_len,
                        n_samples_per_prompt=n_samples_per_prompt,
                        experience_columns=mm_experience_columns,
                        timeout=300.0,
                        experience_consumers=experience_consumers,
                        metrics=metrics,
                        topic="multimodal",
                    )
                    self.tq.add_topic(
                        prompts_num=gbs_train,
                        n_samples_per_prompt=n_samples_per_prompt,
                        experience_columns=mm_experience_columns,
                        timeout=300.0,
                        experience_consumers=experience_consumers,
                        metrics=metrics,
                        topic="multimodal_sampling",
                    )
            else:
                self.tq.add_topic(
                    td_max_len,
                    n_samples_per_prompt,
                    metrics=metrics,
                    experience_columns=experience_columns,
                    timeout=300.0,
                    experience_consumers=experience_consumers,
                )
                if is_multimodal():
                    self.tq.add_topic(
                        prompts_num=td_max_len,
                        n_samples_per_prompt=n_samples_per_prompt,
                        experience_columns=mm_experience_columns,
                        timeout=300.0,
                        experience_consumers=experience_consumers,
                        metrics=metrics,
                        topic="multimodal",
                    )
            self.tq.register_consumer_columns_dict(consumer_columns)
            if is_multimodal():
                self.tq.register_consumer_columns_dict(mm_consumer_columns, "multimodal")
            self.docks.td = SyncTransferQueueAdapter(topic="default_topic", enable_partial_rollout=max_age > 1)
            if should_filter:
                self.docks.sampling_td = SyncTransferQueueAdapter(topic="sampling", enable_partial_rollout=max_age > 1)
            if is_multimodal():
                self.docks.mm_td = SyncTransferQueueAdapter(topic="multimodal", enable_partial_rollout=max_age > 1)
                if should_filter:
                    self.docks.mm_sampling_td = SyncTransferQueueAdapter(topic="multimodal_sampling", enable_partial_rollout=max_age > 1)
        else:
            if should_filter:
                self.docks.td = GRPOTransferDock.remote(
                    max_num_prompt_in_batch,
                    n_samples_per_prompt,
                    max_age=1,
                    GBS_train=max_num_prompt_in_batch,
                    metrics=metrics,
                    addition_columns=addition_columns,
                    addition_consumers=addition_consumers,
                )
                self.docks.sampling_td = GRPOTransferDock.remote(
                    td_max_len,
                    n_samples_per_prompt,
                    max_age=max_age,
                    GBS_train=gbs_train,
                    metrics=metrics,
                    addition_columns=addition_columns,
                    addition_consumers=addition_consumers,
                )
                if is_multimodal():
                    self.docks.mm_td = MMGRPOTransferDock.remote(
                        max_num_prompt_in_batch,
                        n_samples_per_prompt,
                    )
                    self.docks.mm_sampling_td = MMGRPOTransferDock.remote(
                        gbs_train,
                        n_samples_per_prompt,
                    )
            else:
                self.docks.td = GRPOTransferDock.remote(
                    td_max_len,
                    n_samples_per_prompt,
                    max_age=max_age,
                    GBS_train=gbs_train,
                    metrics=metrics,
                    addition_columns=addition_columns,
                    addition_consumers=addition_consumers,
                )
                if is_multimodal():
                    self.docks.mm_td = MMGRPOTransferDock.remote(
                        gbs_train,
                        n_samples_per_prompt,
                    )
        return self.docks

    def clear_main(self, consumer: Optional[str] = None):
        if consumer is None:
            ray.get(self.td.clear.remote())
        else:
            ray.get(self.td.clear.remote(consumer=consumer))

    def clear_sampling(self, consumer: Optional[str] = None):
        if not self.sampling_td:
            return
        if self.strategy == _STRATEGY_TQ and consumer is not None:
            ray.get(self.sampling_td.clear.remote())
        elif consumer is None:
            ray.get(self.sampling_td.clear.remote())
        else:
            ray.get(self.sampling_td.clear.remote(consumer=consumer))

    def clear_mm(self, sampling: bool = False):
        target = self.mm_sampling_td if sampling else self.mm_td
        if target:
            ray.get(target.clear.remote())

    def put_main(self, data_dict, indexes, is_prompt: bool = False):
        ray.get(self.td.put_experience.remote(data_dict=data_dict, indexes=indexes, is_prompt=is_prompt))

    # Backward-compatible name
    def put_experience(self, data_dict, indexes, is_prompt: bool = False):
        self.put_main(data_dict, indexes, is_prompt=is_prompt)

    def put_sampling(self, data_dict, indexes, is_prompt: bool = False):
        if not self.sampling_td:
            return
        ray.get(self.sampling_td.put_experience.remote(data_dict=data_dict, indexes=indexes, is_prompt=is_prompt))

    def put_mm(self, batch, indexes, sampling: bool = False):
        target = self.mm_sampling_td if sampling else self.mm_td
        if not target:
            return
        if self.strategy == _STRATEGY_TQ:
            experience_columns = ray.get(target.get_columns.remote(None))
            batch_mm, indexes_mm = prepare_batch_mm(
                batch,
                experience_columns=experience_columns,
                num_prompts=len(batch["prompts"]),
            )
            ray.get(target.put_experience.remote(
                data_dict=batch_mm,
                indexes=indexes_mm,
                unpad=False,
            ))
        else:
            ray.get(target.put_experience.remote(batch=batch, indexes=indexes))

    def put_mm_data(self, data_dict, indexes, sampling: bool = False):
        target = self.mm_sampling_td if sampling else self.mm_td
        if not target:
            return
        if self.strategy == _STRATEGY_TQ:
            ray.get(target.put_experience.remote(data_dict=data_dict, indexes=indexes, unpad=False))
        else:
            ray.get(target.put_experience.remote(batch=data_dict, indexes=indexes))

    def get_metrics(self):
        return ray.get(self.td.get_metrics.remote())

    def update_metrics(self, key: str, value=None, cumulate: bool = False):
        ray.get(self.td.update_metrics.remote(key, value=value, cumulate=cumulate))

    def prefetch_sampling_index(self, data_num: int):
        if not self.sampling_td:
            return None
        return ray.get(self.sampling_td.prefetch_request_index.remote(data_num))

    def prefetch_main_index(self, data_num: int):
        return ray.get(self.td.prefetch_request_index.remote(data_num))

    def get_cur_index(self):
        return ray.get(self.td.get_cur_index.remote())
