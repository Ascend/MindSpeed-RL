from collections.abc import Mapping
from copy import copy
from typing import Any, Optional, Union
import time

import vllm
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine import EngineCoreRequest


def _is_preprocessed_input(self, prompt: PromptType) -> bool:
    """
    Check if the input is preprocessed format.
    
    Preprocessed format should be:
    [{"prompt_token_ids": [...], "multi_modal_data": {...}}]
    or
    {"prompt_token_ids": [...], "multi_modal_data": {...}}
    """
    if isinstance(prompt, list) and len(prompt) == 1:
        item = prompt[0]
        if isinstance(item, dict):
            return "prompt_token_ids" in item and "multi_modal_data" in item and "image_embeds" in item["multi_modal_data"]
    elif isinstance(prompt, dict):
        return "prompt_token_ids" in prompt and "multi_modal_data" in prompt and "image_embeds" in prompt["multi_modal_data"]
    return False


def _process_preprocessed_input(
        self,
        request_id: str,
        prompt: Union[dict, list],
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> tuple[Optional[str], EngineCoreRequest]:
    """
    Process preprocessed input format and create EngineCoreRequest directly.
    """
    if arrival_time is None:
        arrival_time = time.time()

    # Extract data from preprocessed input
    if isinstance(prompt, list):
        data = prompt[0]
    else:
        data = prompt

    prompt_token_ids = data["prompt_token_ids"]
    multi_modal_data = data["multi_modal_data"]

    # Create MultiModalKwargs from preprocessed multimodal data
    mm_inputs = None
    if multi_modal_data:
        # Convert the preprocessed multimodal data to MultiModalKwargs
        mm_kwargs_data = {}
        for key, value in multi_modal_data.items():
            mm_kwargs_data[key] = value
        mm_inputs = [MultiModalKwargs(mm_kwargs_data)]

        image_token_length = multi_modal_data["image_embeds"].shape[0]

        vision_start_token = self.tokenizer.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        start_pos = None
        for i, token_id in enumerate(prompt_token_ids, 1):
            if token_id == vision_start_token:
                start_pos = i
                break
        
        # Create PlaceholderRange
        from vllm.multimodal.inputs import PlaceholderRange
        mm_placeholders = [PlaceholderRange(
            offset=start_pos,
            length=image_token_length,
            is_embed=None
        )]

        import hashlib
        # Generate hash for preprocessed data
        hash_input = str(mm_kwargs_data).encode('utf-8')
        mm_hash = hashlib.sha256(hash_input).hexdigest()
        mm_hashes = [mm_hash]
        

    # Get EOS token ID
    eos_token_id = self.processor.input_preprocessor.get_eos_token_id(lora_request)

    # Handle sampling params
    sampling_params = params.clone()
    
    # If unset max tokens, then generate up to the max_model_len
    if sampling_params.max_tokens is None:
        sampling_params.max_tokens = (
            self.model_config.max_model_len - len(prompt_token_ids))
    
    # Update sampling params from generation config and tokenizer
    generation_config_fields = self.model_config.try_get_generation_config()
    sampling_params.update_from_generation_config(
        generation_config_fields, eos_token_id)
    sampling_params.update_from_tokenizer(
        self.tokenizer.get_lora_tokenizer(lora_request))

    # Create EngineCoreRequest
    request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_inputs=mm_inputs,
        mm_hashes=mm_hashes,  # No hashing for preprocessed inputs
        mm_placeholders=mm_placeholders,  # No placeholders for preprocessed inputs
        sampling_params=sampling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=None,
        data_parallel_rank=None,
    )

    # Return None for prompt_str since we don't have original prompt string
    return None, request


def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
    # Check if input is preprocessed format
    if self._is_preprocessed_input(prompt):
        # Process preprocessed input directly
        prompt_str, request = self._process_preprocessed_input(
            request_id, prompt, params, arrival_time, lora_request
        )
    else:
        # Process raw inputs into the request (original flow)
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            tokenization_kwargs, trace_headers, prompt_adapter_request,
            priority)

    n = params.n if isinstance(params, SamplingParams) else 1

    if n == 1:
        # Make a new RequestState and queue.
        self.output_processor.add_request(request, prompt_str, None, 0)
        # Add the request to EngineCore.
        self.engine_core.add_request(request)
        return

    # Fan out child requests (for n>1).
    parent_req = ParentRequest(request_id, params)
    for idx in range(n):
        request_id, params = parent_req.get_child_info(idx)
        child_request = request if idx == n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = params

        # Make a new RequestState and queue.
        self.output_processor.add_request(child_request, prompt_str,
                                            parent_req, idx)
        # Add the request to EngineCore.
        self.engine_core.add_request(child_request)


def image_emb_reuse():
    if '0.9.1' in vllm.__version__:
        vllm.v1.engine.llm_engine.LLMEngine._is_preprocessed_input = _is_preprocessed_input
        vllm.v1.engine.llm_engine.LLMEngine._process_preprocessed_input = _process_preprocessed_input
        vllm.v1.engine.llm_engine.LLMEngine.add_request = add_request
    else:
        import warnings
        warnings.warn(
            "ViT reuse feature requires patching vLLM internals to pass and use pre-computed embeddings. "
            "The current implementation is only verified for vLLM version 0.9.1. "
            "This patch may be incompatible or unsafe with other versions. "
            "Please adjust the patch logic accordingly when upgrading vLLM."
        )
