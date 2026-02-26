# Chunked Prefill

## 背景介绍

Chunked Prefill（Splitfuse）特性的目的是将长prompt request分解成更小的块，并在多个forward step中进行调度，只有最后一块的forward完成后才开始这个prompt request的生成。将短prompt request组合以精确填充step的空隙，每个step的计算量基本相等，达到所有请求平均延迟更稳定的目的。

关键行为：

1. 长prompts被分解成更小的块，并在多个迭代中进行调度，只有最后一遍迭代执行输出生成token。
2. 构建batch时，一个prefill块和其余槽位用decode填充，降低仅decode组batch的成本。

其优势主要包括：

* ​**提升效率**​：通过合理组合长短prompt，将长序列预填充阶段分割成 chunk 并 pipeline 化，保持模型高吞吐量运行。
* ​**增强一致性**​：统一前向传递大小，降低延迟波动，使生成频率更稳定。
* **​降低时延，减小OOM：​**通过平衡prefill和decode的计算利用率，降低请求P90_ttft（time to first token）、P90_tpot(time per output token)时延。在短输入、短输出且高并发的场景优势明显，并且可以用于解决 prefill 阶段单 batch 过大导致的 OOM

## 使用方法

1. vLLM脚本中参数max-model-len表示单个请求的最大处理长度，max-num-batched-tokens表示单个推理批次中所有请求的最大tokens数，默认在Chunked Prefill模式下，需满足max-num-batched-tokens ≥ max-model-len，且大于256并能够被256整除，建议该值使用4096、8192甚至更大。
2. 脚本中添加`enable_chunked_prefill: True`。

## 注意事项

* 该特性不能和Prefix Cache(APC)、KV Cache量化特性同时使用。
* Qwen系列模型支持此特性。
* vllm v1 scheduler 默认开启；开启ascend_scheduler_config后，默认关闭。