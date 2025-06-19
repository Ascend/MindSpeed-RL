# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

import os
from contextlib import contextmanager

import gc
import ray
import torch
import torch.distributed
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig


def dummy_compile(*compile_args, **compile_kwargs):
    def decorate(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorate

torch.compile = dummy_compile
torch.jit.script = dummy_compile

from vllm import LLM, SamplingParams
from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.models.base.base_inference_engine import BaseInferEngine
from mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state import initialize_parallel_state
from mindspeed_rl.models.rollout.vllm_adapter.megatron_weight_loaders import (
    load_megatron_weights,
    update_megatron_weight_loader,
    InferParallelConfig
)
from mindspeed_rl.utils import get_tokenizer, is_multimodal


logger = Loggers("vllm_engine")


class VLLMInferEngine(BaseInferEngine):
    def __init__(
            self,
            tokenizer_name_or_path: str,
            train_tensor_parallel_size: int,
            train_pipeline_parallel_size: int,
            train_expert_parallel_size: int,
            train_context_parallel_size: int,
            infer_tensor_parallel_size: int,
            infer_pipeline_parallel_size: int,
            infer_expert_parallel_size: int,
            sampling_config: dict,
            infer_expert_tensor_parallel_size: int = 1,
            prompt_type: str = None,
            prompt_type_path: str = None,
            enable_prefix_caching: bool = False,
            num_scheduler_steps: int = 1,
            max_num_seqs: int = 1,
            max_model_len: int = 2048,
            dtype: str = "bfloat16",
            gpu_memory_utilization: float = 0.5,
            trust_remote_code: bool = True,
            load_format: str = "megatron",
            enforce_eager: bool = False,
            **kwargs
    ):
        """
        Initialize the VLLM inference engine.

        Args:
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            train_tensor_parallel_size (int): Tensor parallel size during training.
            train_pipeline_parallel_size (int): Pipeline parallel size during training.
            train_expert_parallel_size (int): Expert parallel size during training.
            train_context_parallel_size (int): Context parallel size during training.
            infer_tensor_parallel_size (int): Tensor parallel size during inference.
            infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
            infer_expert_parallel_size (int): Expert parallel size during inference.
            sampling_config (dict): Configuration for text generation sampling.
            infer_expert_tensor_parallel_size (int): Expert tensor parallel size during inference.
            enable_prefix_caching (bool): Whether to enable prefix caching.
            num_scheduler_steps (int): Num scheduler steps. Default is 1.
            max_num_seqs (int): Maximum number of sequences to process simultaneously. Default is 1.
            max_model_len (int): Maximum model length (in tokens). Default is 2048.
            dtype (str): Data type for model weights. Default is "bfloat16".
            gpu_memory_utilization (float): GPU memory utilization factor. Default is 0.5.
            trust_remote_code (bool): Whether to trust remote code (e.g., for custom tokenizers).
            **kwargs: Additional keyword arguments.
        """
        # Call the parent class's __init__ method
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            prompt_type=prompt_type,
            prompt_type_path=prompt_type_path,
            train_tensor_parallel_size=train_tensor_parallel_size,
            train_pipeline_parallel_size=train_pipeline_parallel_size,
            train_expert_parallel_size=train_expert_parallel_size,
            train_context_parallel_size=train_context_parallel_size,
            infer_tensor_parallel_size=infer_tensor_parallel_size,
            infer_pipeline_parallel_size=infer_pipeline_parallel_size,
            infer_expert_parallel_size=infer_expert_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code
        )
        # Additional initialization logic for VLLMInferEngine

        torch.compile = dummy_compile
        # vLLM Ascend must be patched in advance
        from vllm_ascend.patch import platform
        from vllm_ascend.patch import worker

        # Initialize sampling parameters from SamplingConfig
        self.sampling_config = sampling_config
        try:
            self.sampling_params = SamplingParams(
                n=sampling_config.get('num_completions', 1),
                logprobs=sampling_config.get('logprobs', 1),
                max_tokens=sampling_config.get('max_tokens', 128),
                top_p=sampling_config.get('top_p', 1.0),
                top_k=sampling_config.get('top_k', 50),
                min_p=sampling_config.get('min_p', 0.0),
                temperature=sampling_config.get('temperature', 0.2),
                detokenize=sampling_config.get('detokenize', False),
                seed=sampling_config.get('seed', None)
            )
        except Exception as e:
            raise ValueError(f"Error creating SamplingParams from dictionary") from e

        self.hf_config = AutoConfig.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code
        )

        self.tokenizer = get_tokenizer(tokenizer_name_or_path,
                                       prompt_type=prompt_type, prompt_type_path=prompt_type_path)
        self.pad_token_id = (
            self.tokenizer.tokenizer.pad_token_id if self.tokenizer.tokenizer.pad_token_id is not None
            else self.tokenizer.tokenizer.eos_token_id)

        # Set up local rank using the helper function
        self.local_rank = get_local_rank()

        # Initialize parallel state if tensor parallel size is specified
        if train_tensor_parallel_size is not None:
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            initialize_parallel_state(
                infer_tensor_model_parallel_size=infer_tensor_parallel_size,
                train_tensor_model_parallel_size=train_tensor_parallel_size,
                infer_pipeline_model_parallel_size=infer_pipeline_parallel_size,
                train_pipeline_model_parallel_size=train_pipeline_parallel_size,
                train_expert_model_parallel_size=train_expert_parallel_size,
                infer_expert_model_parallel_size=infer_expert_parallel_size,
                train_context_model_parallel_size=train_context_parallel_size
            )

        if load_format == "megatron":
            update_megatron_weight_loader()

        # Initialize the LLM engine
        self.llm = LLM(
            model=tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=infer_tensor_parallel_size,
            load_format='dummy' if load_format == 'megatron' else load_format,
            distributed_executor_backend="external_launcher",
            enable_prefix_caching=enable_prefix_caching,
            num_scheduler_steps=num_scheduler_steps,
            dtype=dtype,
            enforce_eager=enforce_eager,
            skip_tokenizer_init=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=self.sampling_params.seed,
            additional_config={
                'expert_tensor_parallel_size': infer_expert_tensor_parallel_size,
                'enable_graph_mode': int(os.environ.get('VLLM_ENABLE_GRAPH_MODE', '0')),
                'ascend_scheduler_config': {},
            }
        )

        self.model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
        self.kv_cache_configs = None

        self.cpu_model = {}
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")

        if load_format == "megatron":
            self.free_cache_engine()
            if os.environ['VLLM_USE_V1'] == '1':
                self._initialize_kv_caches(self.llm.llm_engine.vllm_config)
            self.offload_model_weights()

    from vllm.config import VllmConfig

    def _initialize_kv_caches(self, vllm_config: VllmConfig):

        # Get all kv cache needed by the model
        kv_cache_specs = self.llm.llm_engine.engine_core.engine_core.model_executor.get_kv_cache_specs()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = self.llm.llm_engine.engine_core.engine_core.model_executor.determine_available_memory()

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        self.kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(self.kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == self.kv_cache_configs[0].num_blocks
            for cfg in self.kv_cache_configs
        ])

    def init_cache_engine(self):
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.llm.llm_engine.model_executor.driver_worker.worker
            if not worker.model_runner.kv_caches:
                # v1 使用显式初始化方法
                self.llm.llm_engine.engine_core.engine_core.model_executor.initialize_from_config(
                    self.kv_cache_configs)
        else:
            if self.llm.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
                self.llm.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def free_cache_engine(self):
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.llm.llm_engine.model_executor.driver_worker.worker

            ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context

        else:
            ctx = self.llm.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context
        from vllm.attention import AttentionType

        layer_need_kv_cache = []
        for layer_name in ctx:
            if ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = self.llm.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.llm.llm_engine.model_executor.driver_worker.worker

            # 清理缓存引擎
            worker.model_runner.kv_caches = []
        else:
            self.llm.llm_engine.model_executor.driver_worker.worker.cache_engine = None
            self.llm.llm_engine.model_executor.driver_worker.worker.gpu_cache = None
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        # 多模态kv cache
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.language_model.model.start_layer, self.model.language_model.model.end_layer):
                attn_impl = self.model.language_model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None

        gc.collect()
        torch.cuda.empty_cache()


    def offload_model_weights(self):
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[-1].self_attn, "mla_attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                mla = self.model.model.layers[i].self_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None

    def sync_model_weights(self, params, load_format='megatron'):
        infer_parallel_config = InferParallelConfig(self.infer_tensor_parallel_size, self.infer_pipeline_parallel_size,
                                                    self.infer_expert_parallel_size * self.infer_tensor_parallel_size)
        load_megatron_weights(params,
                              self.model,
                              infer_parallel_config,
                              self.hf_config)
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            self._process_mla()

    def _process_mla(self):
        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            mla = self.model.model.layers[i].self_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
            mla.process_weights_after_loading(None)

    @torch.no_grad()
    def generate_sequences(self, idx_list, **kwargs):
        self.init_cache_engine()
        if is_multimodal():
            images = kwargs.pop("extra_info")
            prompts = [
                {"prompt_token_ids": prompt, "multi_modal_data": {"image": image}}
                for prompt, image in zip(idx_list, images['image'])
            ]
            idx_list = None
        else:
            prompts = None
        with self.update_sampling_params(**kwargs):
            response = self.llm.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False
            )
            outs = self._post_process_outputs(response)
        self.free_cache_engine()
        return outs

    def _post_process_outputs(self, request_outputs):
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs
                if logprobs_dicts is None:
                    continue

                logprob = []
                for logprobs_dict, token_id in zip(logprobs_dicts, output.token_ids):
                    logprob.append(logprobs_dict[token_id].logprob)
                logprobs.append(torch.tensor(logprob))

        output_token_ids = pad_sequence(output_token_ids, batch_first=True,
                                        padding_value=self.pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True,
                                    padding_value=self.pad_token_id)
        return output_token_ids, logprobs

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def chat(self, conversation, sampling_params=None):
        outputs = self.llm.chat(
            conversation,
            sampling_params=sampling_params if sampling_params else self.sampling_params,
            use_tqdm=False)
        return outputs


def get_local_rank() -> int:
    """
    Determine the local rank based on the runtime context.
    - If launched via `torchrun`, the `LOCAL_RANK` environment variable is used.
    - If launched via `ray`, the rank is obtained from the ray runtime context.
    - If neither is available, defaults to 0 (for testing or single-process scenarios).

    Returns:
        int: The local rank of the current process.
    """
    # Check if launched via torchrun (LOCAL_RANK is set)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    # Check if launched via ray
    try:
        # Get the local rank from ray's runtime context
        local_rank_str = ray.get_runtime_context().get_accelerator_ids()["NPU"][0]
        os.environ["LOCAL_RANK"] = local_rank_str
        return int(local_rank_str)

    except Exception as e:
        logger.warning("Warning: Failed to get local rank from ray runtime context. Error: {}".format(e))

    # Default to 0 (for testing or single-process scenarios)
    logger.warning("Warning: Unable to determine local rank. Defaulting to 0.")
    return 0