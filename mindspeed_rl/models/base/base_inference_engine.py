# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from abc import ABC, abstractmethod


class BaseInferEngine(ABC):
    """
    This is the base class for the inference engine.
    It initializes the necessary parameters for the inference process,
    including tokenizer information, parallel sizes during training and inference,
    model length limits, data types, GPU memory utilization, and trust settings for remote code.
    """
    def __init__(
            self,
            tokenizer_name_or_path: str,
            train_tensor_parallel_size: int,
            train_pipeline_parallel_size: int,
            prompt_type: str = None,
            prompt_type_path: str = None,
            train_expert_parallel_size: int = 1,
            infer_tensor_parallel_size: int = 8,
            infer_pipeline_parallel_size: int = 1,
            infer_expert_parallel_size: int = 1,
            max_num_seqs: int = 1,  # Default value set to 1
            max_model_len: int = 2048,  # Default value set to 2048
            dtype: str = "bfloat16",  # Default value set to "bfloat16"
            gpu_memory_utilization: float = 0.5,  # Default value set to 0.5
            trust_remote_code: bool = True
    ):
        """
        Initialize the base inference engine.

        Args:
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            train_tensor_parallel_size (int): Tensor parallel size during training.
            train_pipeline_parallel_size (int): Pipeline parallel size during training.
            train_expert_parallel_size (int): Expert parallel size during training.
            infer_tensor_parallel_size (int): Tensor parallel size during inference.
            infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
            infer_expert_parallel_size (int): Expert parallel size during inference.
            max_num_seqs (int): Maximum number of sequences to process simultaneously. Default is 1.
            max_model_len (int): Maximum model length (in tokens). Default is 2048.
            dtype (str): Data type for model weights. Default is "bfloat16".
            gpu_memory_utilization (float): GPU memory utilization factor. Default is 0.5.
            trust_remote_code (bool): Whether to trust remote code (e.g., for custom tokenizers).
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.prompt_type = prompt_type
        self.prompt_type_path = prompt_type_path
        self.train_tensor_parallel_size = train_tensor_parallel_size
        self.train_pipeline_parallel_size = train_pipeline_parallel_size
        self.train_expert_parallel_size = train_expert_parallel_size
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code


    @abstractmethod
    def init_cache_engine(self):
        pass

    @abstractmethod
    def free_cache_engine(self):
        pass

    @abstractmethod
    def offload_model_weights(self):
        pass

    @abstractmethod
    def sync_model_weights(self, params, load_format='megatron'):
        pass

    @abstractmethod
    def generate_sequences(self,
                           prompts=None,
                           sampling_params=None,
                           prompt_token_ids=None,
                           use_tqdm=None,
                           **kwargs):
        pass
