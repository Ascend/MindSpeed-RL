# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

import os
import uuid
from contextlib import contextmanager

import gc
import ray
import torch
import torch.distributed
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig

from vllm import LLM, SamplingParams
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
    """VLLM inference engine for distributed model inference.

    This class extends BaseInferEngine to provide vLLM-based inference capabilities
    with support for various parallelism strategies including tensor, pipeline, and
    expert parallelism. It handles model weight loading, KV cache management, and
    text generation for reinforcement learning training workflows.

    Attributes:
        sampling_config (dict): Configuration dictionary for text generation sampling.
        sampling_params (SamplingParams): vLLM sampling parameters instance.
        hf_config (AutoConfig): HuggingFace model configuration.
        tokenizer (Tokenizer): Tokenizer instance for text encoding/decoding.
        pad_token_id (int): Token ID used for padding sequences.
        local_rank (int): Local rank of the current process.
        llm (LLM): vLLM LLM engine instance.
        engine: Reference to the underlying vLLM engine.
        model: Reference to the underlying model instance.
        fused_moe: Fused MoE layer for expert parallel load balancing.
        eplb_map (torch.Tensor): Expert load balancer mapping tensor.
        global_redundant_expert_num (int): Number of global redundant experts.
        infer_local_num_experts (int): Number of local experts for inference.
        cpu_model (dict): CPU buffer for model weights offload.
    """

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
            prompt_type: str = None,
            prompt_type_path: str = None,
            enable_prefix_caching: bool = False,
            num_scheduler_steps: int = 1,
            max_num_seqs: int = 1,
            max_model_len: int = 2048,
            max_num_batched_tokens: int = 2048,
            dtype: str = "bfloat16",
            gpu_memory_utilization: float = 0.5,
            trust_remote_code: bool = False,
            load_format: str = "megatron",
            enforce_eager: bool = False,
            torchair_graph: bool = False,
            ascend_scheduler_config_enabled: bool = True,
            limit_mm_image_per_prompt: int = 1,
            limit_mm_video_per_prompt: int = 0,
            enable_expert_parallel: bool = False,
            expert_map_path: str = None,
            eplb_token_collects: bool = False,
            eplb_token_save_path: str = None,
            **kwargs
    ):
        """Initialize the VLLM inference engine.

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
            prompt_type (str, optional): Type of prompt template to use. Defaults to None.
            prompt_type_path (str, optional): Path to custom prompt type configuration. Defaults to None.
            enable_prefix_caching (bool, optional): Whether to enable prefix caching. Defaults to False.
            num_scheduler_steps (int, optional): Number of scheduler steps. Defaults to 1.
            max_num_seqs (int, optional): Maximum number of sequences to process simultaneously. Defaults to 1.
            max_model_len (int, optional): Maximum model length in tokens. Defaults to 2048.
            max_num_batched_tokens (int, optional): Maximum number of batched tokens. Defaults to 2048.
            dtype (str, optional): Data type for model weights. Defaults to "bfloat16".
            gpu_memory_utilization (float, optional): GPU memory utilization factor. Defaults to 0.5.
            trust_remote_code (bool, optional): Whether to trust remote code for custom tokenizers. Defaults to False.
            load_format (str, optional): Format for loading model weights. Defaults to "megatron".
            enforce_eager (bool, optional): Whether to enforce eager execution mode. Defaults to False.
            torchair_graph (bool, optional): Whether to enable torchair graph compilation. Defaults to False.
            ascend_scheduler_config_enabled (bool, optional): Whether to enable Ascend scheduler config. Defaults to True.
            limit_mm_image_per_prompt (int, optional): Maximum images per prompt for multimodal. Defaults to 1.
            limit_mm_video_per_prompt (int, optional): Maximum videos per prompt for multimodal. Defaults to 0.
            enable_expert_parallel (bool, optional): Whether to enable expert parallelism. Defaults to False.
            expert_map_path (str, optional): Path to expert mapping configuration. Defaults to None.
            eplb_token_collects (bool, optional): Whether to collect tokens for EPLB. Defaults to False.
            eplb_token_save_path (str, optional): Path to save EPLB token statistics. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
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
            trust_remote_code=trust_remote_code,
            enable_expert_parallel=enable_expert_parallel,
        )

        # Apply vLLM Ascend patches
        from vllm_ascend.patch import platform
        from vllm_ascend.patch import worker
        from mindspeed_rl.models.rollout.vllm_adapter import engine_core

        # Configure Expert Parallel Load Balancing (EPLB)
        from mindspeed_rl.models.rollout.vllm_adapter.fused_moe import set_EPLB_args
        from mindspeed_rl.models.rollout.vllm_adapter import fused_moe
        set_EPLB_args(eplb_token_collects, eplb_token_save_path)

        # Initialize sampling parameters from configuration
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
            raise ValueError("Error creating SamplingParams from dictionary") from e

        # Load model configuration and tokenizer
        self.hf_config = AutoConfig.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code
        )

        self.tokenizer = get_tokenizer(
            tokenizer_name_or_path,
            prompt_type=prompt_type,
            prompt_type_path=prompt_type_path
        )
        self.pad_token_id = (
            self.tokenizer.tokenizer.pad_token_id
            if self.tokenizer.tokenizer.pad_token_id is not None
            else self.tokenizer.tokenizer.eos_token_id
        )

        # Set up local rank for distributed training
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

        # Update weight loader for Megatron format
        if load_format == "megatron":
            update_megatron_weight_loader()

        # Configure multimodal and scheduler settings
        limit_mm_per_prompt_dict = {}
        ascend_scheduler_config = {"enabled": ascend_scheduler_config_enabled}
        graph_batch_sizes = [max_num_seqs] if torchair_graph else []

        if is_multimodal():
            if limit_mm_image_per_prompt > 0:
                limit_mm_per_prompt_dict['image'] = limit_mm_image_per_prompt
            if limit_mm_video_per_prompt > 0:
                limit_mm_per_prompt_dict['video'] = limit_mm_video_per_prompt
            ascend_scheduler_config = {}
            graph_batch_sizes = []

        # Initialize the vLLM LLM engine
        self.llm = LLM(
            model=tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=infer_tensor_parallel_size,
            load_format='dummy' if load_format == 'megatron' else load_format,
            distributed_executor_backend="external_launcher",
            enable_prefix_caching=enable_prefix_caching,
            dtype=dtype,
            enforce_eager=enforce_eager,
            skip_tokenizer_init=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=self.sampling_params.seed,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_expert_parallel=enable_expert_parallel,
            limit_mm_per_prompt=limit_mm_per_prompt_dict,
            additional_config={
                "torchair_graph_config": {
                    "enabled": torchair_graph,
                    "use_cached_graph": False,
                    "graph_batch_sizes_init": False,
                    "graph_batch_sizes": graph_batch_sizes,
                },
                "ascend_scheduler_config": ascend_scheduler_config,
                "refresh": True,
                "expert_map_path": expert_map_path,
            }
        )

        self.engine = self.llm.llm_engine
        self.model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()

        # Extract EPLB-related attributes from model
        if expert_map_path is not None:
            self.fused_moe = self.model.model.layers[-1].mlp.experts
            self.eplb_map = self.fused_moe.expert_load_balancer.expert_map_tensor
            self.global_redundant_expert_num = self.fused_moe.global_redundant_expert_num
            self.infer_local_num_experts = self.fused_moe.local_num_experts
        else:
            self.eplb_map = None
            self.global_redundant_expert_num = 0
            self.infer_local_num_experts = -1

        # Initialize CPU buffer for model weights offload
        self.cpu_model = {}
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")

        # Offload weights if using Megatron format
        if load_format == "megatron":
            self.free_cache_engine()
            self.offload_model_weights()

    def init_cache_engine(self):
        """Initialize the KV cache engine.

        Initializes the key-value cache engine for vLLM inference.
        Supports both v1 and legacy vLLM versions with different initialization methods.
        """
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.llm.llm_engine.model_executor.driver_worker.worker
            if not worker.model_runner.kv_caches:
                # v1 uses explicit initialization method
                self.llm.llm_engine.engine_core.engine_core.model_executor.initialize_from_config(
                    self.llm.llm_engine.engine_core.engine_core.kv_cache_configs
                )
                self.llm.llm_engine.reset_prefix_cache()
        else:
            if self.llm.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
                self.llm.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def free_cache_engine(self):
        """Free the KV cache engine and release memory.

        Releases all KV cache resources including cache engine, GPU cache,
        and attention implementation caches. Performs garbage collection
        and empties CUDA cache to reclaim memory.
        """
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.llm.llm_engine.model_executor.driver_worker.worker
            ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context
        else:
            ctx = self.llm.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context

        from vllm.attention import AttentionType

        # Identify layers that need KV cache
        layer_need_kv_cache = []
        for layer_name in ctx:
            if hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type in (
                AttentionType.DECODER, AttentionType.ENCODER_DECODER
            ):
                layer_need_kv_cache.append(layer_name)

        # Clear KV cache for each layer
        pipeline_parallel_size = self.llm.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache

        # Clear cache engine based on vLLM version
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.llm.llm_engine.model_executor.driver_worker.worker

            for _, kv_caches_i in enumerate(worker.model_runner.kv_caches):
                for _, kv_caches_i_j in enumerate(kv_caches_i):
                    kv_caches_i_j.untyped_storage().resize_(0)

            worker.model_runner.kv_caches = []
        else:
            self.llm.llm_engine.model_executor.driver_worker.worker.cache_engine = None
            self.llm.llm_engine.model_executor.driver_worker.worker.gpu_cache = None

        # Clear attention implementation caches for language models
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        # Clear attention implementation caches for multimodal models
        elif hasattr(self.model, 'language_model') and hasattr(
            self.model.language_model.model.layers[0].self_attn, "attn"
        ):
            for i in range(
                self.model.language_model.model.start_layer,
                self.model.language_model.model.end_layer
            ):
                attn_impl = self.model.language_model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None

        gc.collect()
        torch.cuda.empty_cache()

    def offload_model_weights(self):
        """Offload model weights to CPU memory.

        Transfers model parameters from GPU to CPU to free GPU memory.
        Also clears MLA (Multi-Head Latent Attention) cached weights if present.
        """
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]

        # Clear MLA cached weights
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[-1].self_attn, "mla_attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                mla = self.model.model.layers[i].self_attn.mla_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None

    def sync_model_weights(self, params, load_format='megatron'):
        """Synchronize model weights from training to inference engine.

        Loads Megatron-format weights into the inference model and processes
        MLA weights if applicable.

        Args:
            params: Model parameters to load.
            load_format (str, optional): Format for loading weights. Defaults to 'megatron'.
        """
        infer_parallel_config = InferParallelConfig(
            self.infer_tensor_parallel_size,
            self.infer_pipeline_parallel_size,
            self.infer_expert_parallel_size * self.infer_tensor_parallel_size,
            self.infer_local_num_experts
        )
        load_megatron_weights(
            params,
            self.model,
            infer_parallel_config,
            self.hf_config
        )

        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            self._process_mla()

    def _process_mla(self):
        """Process MLA weights after loading.

        Clears cached MLA weights and reprocesses them after model weight loading.
        This is necessary for Multi-Head Latent Attention mechanism initialization.
        """
        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            mla = self.model.model.layers[i].self_attn.mla_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
            mla.process_weights_after_loading(None)

    @torch.no_grad()
    def generate_sequences(self, idx_list, **kwargs):
        """Generate sequences from input token IDs.

        Performs inference to generate text sequences based on input prompts.
        Supports both text-only and multimodal inputs (images/videos).

        Args:
            idx_list (list): List of input token ID sequences.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            tuple: (output_token_ids, logprobs) where:
                - output_token_ids (torch.Tensor): Generated token IDs.
                - logprobs (torch.Tensor): Log probabilities of generated tokens.
        """
        self.init_cache_engine()

        if is_multimodal():
            images = kwargs.pop("extra_info")
            prompts = []

            if "vit_embeds" in images:
                # Process image embeddings
                for prompt, image_embeds, grid_thw in zip(
                    idx_list, images['vit_embeds'], images['image_grid_thw']
                ):
                    prompt_data = {
                        "prompt_token_ids": prompt,
                        "multi_modal_data": {
                            "image_embeds": image_embeds,
                            "image_grid_thw": grid_thw
                        }
                    }
                    prompts.append(prompt_data)

                # Handle video inputs by replacing video token IDs with image token IDs
                if torch.sum(images['video_num']).item() > 0:
                    from megatron.training import get_args
                    for p in prompts:
                        p['prompt_token_ids'] = list(map(
                            lambda x: get_args().mm.model.image_token_id
                            if x == get_args().mm.model.video_token_id else x,
                            p['prompt_token_ids']
                        ))
            else:
                # Process raw images
                if torch.sum(images['image_num']).item() > 0:
                    for prompt, image in zip(idx_list, images['image']):
                        prompt_data = {
                            "prompt_token_ids": prompt,
                            "multi_modal_data": {"image": image}
                        }
                        prompts.append(prompt_data)
                else:
                    # Process videos
                    for prompt, video, fps in zip(prompts, images['video'], images['video_fps']):
                        prompt = {
                            "prompt_token_ids": prompt,
                            "multi_modal_data": {"video": video},
                            'mm_processor_kwargs': {'fps': fps.squeeze().tolist()}
                        }
                        prompts.append(prompt)

            idx_list = None
        else:
            # Decode token IDs to text prompts for non-multimodal inputs
            prompts = [
                self.tokenizer.tokenizer.decode(p, skip_special_tokens=True)
                for p in idx_list
            ]

        with self.update_sampling_params(**kwargs):
            response = self.llm.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            outs = self._post_process_outputs(response)

        self.free_cache_engine()
        return outs

    @torch.no_grad()
    def async_generate_sequences(self, idx_list, indexes, stop_singal_func=None, **kwargs):
        """Asynchronously generate sequences with streaming output.

        Generates sequences incrementally, yielding results as they become available.
        Supports early stopping via signal function.

        Args:
            idx_list (list): List of input token ID sequences.
            indexes (list): List of request indices corresponding to inputs.
            stop_singal_func (callable, optional): Function to check for stop signal.
                Should return truthy value to trigger stopping. Defaults to None.
            **kwargs: Additional sampling parameters to override defaults.

        Yields:
            tuple: ((prompt_ids, response_ids), index) where:
                - prompt_ids (list): List containing input token IDs tensor.
                - response_ids (tuple): Generated output IDs and logprobs.
                - index (int): Request index.
        """
        STOP_SIGNAL = None

        with self.update_sampling_params(**kwargs):
            # Add requests to the engine
            for i, prompt_token_ids in enumerate(idx_list):
                request_id = f"req_{indexes[i]}_{uuid.uuid4().hex[:6]}"
                self.engine.add_request(
                    request_id=request_id,
                    prompt={"prompt_token_ids": prompt_token_ids},
                    params=self.sampling_params
                )

            count = 0
            while self.engine.has_unfinished_requests():
                count += 1
                if stop_singal_func is not None and count % 20 == 0:
                    STOP_SIGNAL = stop_singal_func()

                step_outputs = self.engine.step()
                for output in step_outputs:
                    if output.finished or STOP_SIGNAL:
                        request_id = output.request_id
                        index = int(request_id.split("_")[1])
                        prompt_ids = [torch.tensor(idx_list[indexes.index(index)])]
                        index = [index]
                        response_ids = self._post_process_outputs([output])
                        yield (prompt_ids, *response_ids), index

                    if STOP_SIGNAL:
                        self.engine.abort_request([request_id])

    def _post_process_outputs(self, request_outputs):
        """Post-process generation outputs.

        Extracts token IDs and log probabilities from vLLM request outputs
        and pads them to create uniform tensors.

        Args:
            request_outputs (list): List of RequestOutput from vLLM.

        Returns:
            tuple: (output_token_ids, logprobs) where:
                - output_token_ids (torch.Tensor): Padded output token IDs.
                - logprobs (torch.Tensor): Padded log probabilities.
        """
        output_token_ids = []
        logprobs = []

        for request_output in request_outputs:
            outputs = request_output.outputs
            for output in outputs:
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs

                if logprobs_dicts is None:
                    continue

                logprob = []
                for logprobs_dict, token_id in zip(logprobs_dicts, output.token_ids):
                    logprob.append(logprobs_dict[token_id].logprob)
                logprobs.append(torch.tensor(logprob))

        output_token_ids = pad_sequence(
            output_token_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        if len(logprobs) > 0:
            logprobs = pad_sequence(
                logprobs,
                batch_first=True,
                padding_value=self.pad_token_id
            )

        return output_token_ids, logprobs

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Context manager for temporarily updating sampling parameters.

        Allows temporary override of sampling parameters for a single generation
        call, automatically restoring original values afterwards.

        Args:
            **kwargs: Sampling parameter names and values to override.

        Yields:
            None: Control is yielded to the wrapped code block.
        """
        old_sampling_params_args = {}

        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield

        # Restore original sampling parameters
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def chat(self, conversation, sampling_params=None):
        """Conduct a chat conversation using the model.

        Args:
            conversation (list): List of conversation messages.
            sampling_params (SamplingParams, optional): Custom sampling parameters.
                Defaults to None, using self.sampling_params.

        Returns:
            list: Chat outputs from the model.
        """
        outputs = self.llm.chat(
            conversation,
            sampling_params=sampling_params if sampling_params else self.sampling_params,
            use_tqdm=False
        )
        return outputs


def get_local_rank() -> int:
    """Determine the local rank of the current process.

    Detects the local rank based on the runtime environment:
    - If launched via torchrun, uses LOCAL_RANK environment variable.
    - If launched via Ray, obtains rank from Ray runtime context.
    - Defaults to 0 for single-process or testing scenarios.

    Returns:
        int: The local rank of the current process.
    """
    # Check if launched via torchrun (LOCAL_RANK is set)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    # Check if launched via Ray
    try:
        local_rank_str = ray.get_runtime_context().get_accelerator_ids()["NPU"][0]
        os.environ["LOCAL_RANK"] = local_rank_str
        return int(local_rank_str)
    except Exception as e:
        logger.warning("Failed to get local rank from ray runtime context. Error: {}".format(e))

    # Default to 0 for testing or single-process scenarios
    logger.warning("Unable to determine local rank. Defaulting to 0.")
    return 0