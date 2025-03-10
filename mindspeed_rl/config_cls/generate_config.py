# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.config_cls.base_config import BaseConfig


class GenerateConfig(BaseConfig):
    """
    vLLM configuration class.
    """

    def __init__(self, config_dict):
        """
        Initialize model configuration from the provided config dictionary.
        All instance attributes are initialized using the dictionary keys.

        :param config_dict: Dictionary containing the configuration parameters.
                            If None, default values will be used for all attributes.
        tokenizer_name_or_path: Path or name of the tokenizer. Default is "/path/to/tokenizer".
        trust_remote_code: Whether to trust remote code (e.g., for custom tokenizers). Default is True.

        infer_tensor_parallel_size: Tensor parallel size during inference. Default is 8.
        infer_pipeline_parallel_size: Pipeline parallel size during inference. Default is 1.
        infer_expert_parallel_size: Expert parallel size during inference. Default is 1.
        
        max_num_seqs: Maximum number of sequences to process simultaneously. Default is 256.
        max_model_len: Maximum model length (in tokens). Default is 2048.
        dtype: Data type for model weights. Default is "bfloat16".
        gpu_memory_utilization: GPU memory utilization factor. Default is 0.5.
        
        sampling_config: Configuration for text generation sampling. Default values are set for various sampling parameters.
            - num_completions: The number of independent completions to generate for each input prompt. Default is 1.
            - logprobs: The number of top tokens to return log probabilities for. Default is 1.
            - max_tokens: The maximum number of tokens to generate in the output. Default is 128.
            - best_of: The number of candidate completions to generate internally before returning the best one. Default is 2.
            - top_p: The cumulative probability threshold for nucleus sampling. Default is 1.0.
            - top_k: The number of highest - probability tokens to consider for sampling. Default is 50.
            - min_p: The minimum probability threshold for token selection. Default is 0.0.
            - temperature: Controls the randomness of predictions by scaling the logits before applying softmax. Default is 0.2.
            - detokenize: Whether to convert the generated tokens back into a human - readable string. Default is False.
        """

        self.tokenizer_name_or_path = "/path/to/tokenizer"
        self.trust_remote_code = True
        
        self.infer_tensor_parallel_size = 8
        self.infer_pipeline_parallel_size = 1
        self.infer_expert_parallel_size = 1
        
        self.max_num_seqs = 1
        self.max_model_len = 2048

        self.dtype = "bfloat16"
        self.gpu_memory_utilization = 0.5


        self.sampling_config = {
            "num_completions": 1,  # 每个输入提示生成的独立完成项数量
            "logprobs": 1,  # 返回的 top token 的对数概率数量
            "max_tokens": 128,  # 生成输出的最大 token 数量
            "best_of": 2,  # 内部生成候选完成项的数量，从中选择最佳的一个
            "top_p": 1.0,  # 核采样的累积概率阈值
            "top_k": 50,  # 采样时考虑的最高概率 token 的数量
            "min_p": 0.0,  # token 选择的最小概率阈值
            "temperature": 0.2,  # 控制预测随机性的温度参数
            "detokenize": False  # 是否将生成的 token 转换回可读字符串
        }

        if config_dict is not None:
            self.update(config_dict)