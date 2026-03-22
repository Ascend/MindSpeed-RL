# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from abc import ABC, abstractmethod


class BaseInferEngine(ABC):
    """Base class for the inference engine.

    This class initializes the necessary parameters for the inference process,
    including tokenizer information, parallel sizes during training and inference,
    model length limits, data types and trust settings for remote code.
    """

    def __init__(
            self,
            tokenizer_name_or_path: str,
            train_tensor_parallel_size: int,
            train_pipeline_parallel_size: int,
            prompt_type: str = None,
            prompt_type_path: str = None,
            train_expert_parallel_size: int = 1,
            train_context_parallel_size: int = 1,
            infer_tensor_parallel_size: int = 8,
            infer_pipeline_parallel_size: int = 1,
            infer_expert_parallel_size: int = 1,
            max_num_seqs: int = 1,
            max_model_len: int = 2048,
            dtype: str = "bfloat16",
            gpu_memory_utilization: float = 0.5,
            trust_remote_code: bool = False,
            enable_expert_parallel: bool = False,
    ):
        """Initialize the base inference engine.

        Args:
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            train_tensor_parallel_size (int): Tensor parallel size during training.
            train_pipeline_parallel_size (int): Pipeline parallel size during training.
            prompt_type (str, optional): Template name for constructing prompts. Defaults to None.
            prompt_type_path (str, optional): Path to the JSON file containing prompt templates. Defaults to None.
            train_expert_parallel_size (int, optional): Expert parallel size during training. Defaults to 1.
            train_context_parallel_size (int, optional): Context parallel size during training. Defaults to 1.
            infer_tensor_parallel_size (int, optional): Tensor parallel size during inference. Defaults to 8.
            infer_pipeline_parallel_size (int, optional): Pipeline parallel size during inference. Defaults to 1.
            infer_expert_parallel_size (int, optional): Expert parallel size during inference. Defaults to 1.
            max_num_seqs (int, optional): Maximum number of sequences to process simultaneously. Defaults to 1.
            max_model_len (int, optional): Maximum model length (in tokens). Defaults to 2048.
            dtype (str, optional): Data type for model weights. Defaults to "bfloat16".
            gpu_memory_utilization (float, optional): memory utilization factor. Defaults to 0.5.
            trust_remote_code (bool, optional): Whether to trust remote code (e.g., for custom tokenizers). Defaults to False.
            enable_expert_parallel (bool, optional): Whether to enable expert parallel. Defaults to False.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.prompt_type = prompt_type
        self.prompt_type_path = prompt_type_path
        self.train_tensor_parallel_size = train_tensor_parallel_size
        self.train_pipeline_parallel_size = train_pipeline_parallel_size
        self.train_expert_parallel_size = train_expert_parallel_size
        self.train_context_parallel_size = train_context_parallel_size
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.enable_expert_parallel = enable_expert_parallel

    @abstractmethod
    def init_cache_engine(self):
        """Initialize the cache engine for inference.

        This method should be implemented by subclasses to initialize
        the cache engine for efficient inference.
        """
        pass

    @abstractmethod
    def free_cache_engine(self):
        """Free the cache engine resources.

        This method should be implemented by subclasses to release
        the cache engine resources.
        """
        pass

    @abstractmethod
    def offload_model_weights(self):
        """Offload model weights to CPU or other storage.

        This method should be implemented by subclasses to offload
        model weights from NPU memory.
        """
        pass

    @abstractmethod
    def sync_model_weights(self, params, load_format='megatron'):
        """Synchronize model weights from external source.

        Args:
            params: Model parameters to synchronize.
            load_format (str, optional): Format for loading weights. Defaults to 'megatron'.
        """
        pass

    @abstractmethod
    def generate_sequences(self,
                           prompts=None,
                           sampling_params=None,
                           prompt_token_ids=None,
                           use_tqdm=None,
                           **kwargs):
        """Generate sequences based on input prompts.

        Args:
            prompts: Input prompts for generation.
            sampling_params: Parameters for controlling sampling behavior.
            prompt_token_ids: Token IDs of the input prompts.
            use_tqdm: Whether to use tqdm progress bar.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Generated sequences based on the input prompts.
        """
        pass