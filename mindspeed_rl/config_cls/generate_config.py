# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.config_cls.base_config import BaseConfig


class GenerateConfig(BaseConfig):
    """Generate configuration class for model generation settings.

    This class manages generation parameters including parallel configuration,
    tokenizer settings, memory management, and sampling parameters for inference.

    Attributes:
        limit_mm_image_per_prompt (int): Maximum number of images per prompt in multi-image scenarios.
        limit_mm_video_per_prompt (int): Maximum number of videos per prompt in multi-video scenarios.
        data_parallel_size (int): Data parallel size for rollout.
        tokenizer_name_or_path (str): Path or name of the tokenizer.
        trust_remote_code (bool): Whether to trust remote code (e.g., for custom tokenizers).
        eplb_token_collects (bool): Whether to collect the number of tokens from each expert for EPLB.
        eplb_token_save_path (str): Save path of the token counts for each expert.
        expert_map_path (str): Path of expert_map in EPLB. When None, EPLB is used.
        infer_tensor_parallel_size (int): Tensor parallel size during inference.
        infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
        infer_expert_parallel_size (int): Expert parallel size during inference.
        max_num_seqs (int): Maximum number of sequences to process simultaneously.
        max_model_len (int): Maximum model length (in tokens).
        max_num_batched_tokens (int): The maximum number of tokens model can run in a single batch.
        dtype (str): Data type for model weights (e.g., "bfloat16", "float16").
        gpu_memory_utilization (float): GPU memory utilization factor.
        offload_train_optimizer (bool): Whether to offload training optimizer state to CPU.
        offload_train_grad (bool): Whether to offload training gradients to CPU.
        offload_train_param (bool): Whether to offload training parameters to CPU.
        enable_prefix_caching (bool): Whether to enable prefix caching for generation.
        num_scheduler_steps (int): Number of scheduler steps for generation.
        enforce_eager (bool): Whether to always use eager-mode PyTorch (disable ACL graph).
        torchair_graph (bool): Whether to enable TorchAir graph optimization.
        enable_expert_parallel (bool): Whether to enable expert parallel computation for MoE layers.
        ascend_scheduler_config_enabled (bool): Whether to enable ascend scheduler config.
        sampling_config (dict): Configuration for text generation sampling parameters including:
            - logprobs (int): Number of top tokens to return log probabilities for.
            - max_tokens (int): Maximum number of tokens to generate.
            - top_p (float): Cumulative probability threshold for nucleus sampling.
            - top_k (int): Number of highest-probability tokens to consider.
            - min_p (float): Minimum probability threshold for token selection.
            - temperature (float): Controls randomness of predictions.
            - detokenize (bool): Whether to convert tokens back to readable string.
            - seed (int): Random seed for sampling.
    """

    def __init__(self, config_dict):
        """Initialize GenerateConfig with configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing the configuration parameters.
                If None, default values will be used for all attributes.
        """
        self.limit_mm_image_per_prompt = 1
        self.limit_mm_video_per_prompt = 0

        self.data_parallel_size = None
        self.tokenizer_name_or_path = "/path/to/tokenizer"
        self.trust_remote_code = False

        self.eplb_token_collects = False
        self.eplb_token_save_path = "./"
        self.expert_map_path = None

        self.infer_tensor_parallel_size = 8
        self.infer_pipeline_parallel_size = 1
        self.infer_expert_parallel_size = 1

        self.max_num_seqs = 1
        self.max_model_len = 2048
        self.max_num_batched_tokens = 2048
        self.dtype = "bfloat16"

        self.gpu_memory_utilization = 0.5
        self.offload_train_optimizer = False
        self.offload_train_grad = False
        self.offload_train_param = False

        self.enable_prefix_caching = False
        self.num_scheduler_steps = 1
        self.enforce_eager = True
        self.torchair_graph = False
        self.enable_expert_parallel = False
        self.ascend_scheduler_config_enabled = True

        self.sampling_config = {
            "logprobs": 1,
            "max_tokens": 128,
            "top_p": 1.0,
            "top_k": 50,
            "min_p": 0.0,
            "temperature": 0.2,
            "detokenize": False,
            "seed": None
        }

        if config_dict.get("sampling_config") is not None:
            for key, _ in config_dict["sampling_config"].items():
                if key not in self.sampling_config:
                    raise ValueError(f"The key: {key} is missing, causing the setup to fail. Please check."
                            f" If necessary, register it in the config file.")

        self.update(config_dict)