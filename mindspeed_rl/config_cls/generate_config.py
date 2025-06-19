# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.config_cls.base_config import BaseConfig


class GenerateConfig(BaseConfig):
    """
    Generate configuration class.
    Initialize model configuration from the provided config dictionary.
    All instance attributes are initialized using the dictionary keys.

    :param config_dict: Dictionary containing the configuration parameters.
                        If None, default values will be used for all attributes.
    data_parallel_size: data parallel size for rollout (default: None)
    tokenizer_name_or_path: Path or name of the tokenizer. Default is "/path/to/tokenizer".
    trust_remote_code: Whether to trust remote code (e.g., for custom tokenizers). Default is True.

    infer_tensor_parallel_size: Tensor parallel size during inference. Default is 8.
    infer_pipeline_parallel_size: Pipeline parallel size during inference. Default is 1.
    infer_expert_parallel_size: Expert parallel size during inference. Default is 1.

    max_num_seqs: Maximum number of sequences to process simultaneously. Default is 256.
    max_model_len: Maximum model length (in tokens). Default is 2048.
    dtype: Data type for model weights. Default is "bfloat16".
    gpu_memory_utilization: GPU memory utilization factor. Default is 0.5.

    enforce_eager: Whether to always use eager-mode PyTorch. If True, we will disable ACL graph and always execute the model in eager mode. 
                   If False, we will use ACL graph and eager execution in hybrid for maximal performance and flexibility. 

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

    def __init__(self, config_dict):
        self.data_parallel_size = None
        # 设置 tokenizer 的名称或路径，默认指向一个示例路径，可根据实际情况修改
        self.tokenizer_name_or_path = "/path/to/tokenizer"
        # 是否信任远程代码，例如用于自定义 tokenizer，默认为 True
        self.trust_remote_code = True
        
        # 推理时的张量并行大小，默认为 8
        self.infer_tensor_parallel_size = 8

        # 推理时的流水线并行大小，默认为 1
        self.infer_pipeline_parallel_size = 1

        # 推理时的专家并行大小，默认为 1
        self.infer_expert_parallel_size = 1

        # 最大可处理的序列数量，默认为 1
        self.max_num_seqs = 1

        # 模型的最大长度（以 token 为单位），默认为 2048
        self.max_model_len = 2048

        self.max_num_batched_tokens = 2048

        # 模型权重的数据类型，默认为 bfloat16
        self.dtype = "bfloat16"

        # GPU 内存的利用率，默认为 0.5
        self.gpu_memory_utilization = 0.5
        self.offload_train_optimizer = False
        self.offload_train_grad = False
        self.offload_train_param = False

        self.enable_prefix_caching = False
        self.num_scheduler_steps = 1
        self.enforce_eager = False

        # 采样配置的默认值，用于生成文本时的采样策略设置
        self.sampling_config = {
            "logprobs": 1,  # 返回的 top token 的对数概率数量
            "max_tokens": 128,  # 生成输出的最大 token 数量
            "top_p": 1.0,  # 核采样的累积概率阈值
            "top_k": 50,  # 采样时考虑的最高概率 token 的数量
            "min_p": 0.0,  # token 选择的最小概率阈值
            "temperature": 0.2,  # 控制预测随机性的温度参数
            "detokenize": False,  # 是否将生成的 token 转换回可读字符串
            "seed": None # 随机种子
        }

        if config_dict.get("sampling_config") is not None:
            for key, _ in config_dict["sampling_config"].items():
                if key not in self.sampling_config:
                    raise ValueError(f"The key: {key} is missing, causing the setup to fail. Please check."
                            f" If necessary, register it in the config file.")    

        # 如果提供了配置字典，则更新默认值
        self.update(config_dict)
