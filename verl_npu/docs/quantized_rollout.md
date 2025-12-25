## 在线量化权重：

利用 [Flash-RL](https://github.com/yaof20/Flash-RL) 工具，修改推理后端，生成 INT8 和 FP8 的 RL rollout。下文以 Qwen2.5-7B int8 为例，在 NPU 上跑通端到端功能。

### 使用步骤：

#### 1、安装包：

```
pip install flash-llm-rl # need to be installed in all nodes in multi-node training
```

#### 2、打patch

安装 FlashRL 后，默认采用自动 patch，推荐改用手动方式，减少过程中的错误：

1. 在 `verl/verl/__init__.py` 文件中添加 `import flash_rl`；
2. 在 shell 脚本中添加 `flashrl cleanup`，这将禁用自动 patch；

#### 3、生成性能分析文件

具体来说，profile 文件会比较 bf16 模型和 int8 模型，以确定如何对更新后的模型执行在线量化：

```
flashrl profile -m Qwen/Qwen2.5-7B -q RedHatAI/Qwen2.5-7B-quantized.w8a8 -o ${PROFILE_PATH:-"$HOME/profile.7b.pt"} --fn int8
```

`-m` 参数后是 bf16 模型路径，`-q` 参数后是 int8 模型路径，`-o` 参数后是生成文件路径；
[RedHatAI](https://huggingface.co/RedHatAI/collections) 提供了各种量化模型；

#### 4、生成配置文件

通过以下命令生成 yaml 配置文件，供 patch 程序使用：

```
flashrl setup -m RedHatAI/Qwen2.5-7B-quantized.w8a8 -p $HOME/profile.7b.pt --fn int8 -o ${CONFIG_PATH:-"$HOME/.flashrl_config.7b.yaml"}
```

`-m` 参数后是 int8 模型路径，`-p` 参数后是 profile 文件路径，`-o` 参数后是生成文件路径；

（可选）为了缩小 rollout 生成和梯度计算之间的差距，FlashRL 提供了在 DP 工作线程间以混合方式进行 16 位和 8 位 rollout 生成的功能。具体来说，运行以下命令会将第二个配置附加到现有的 yaml 配置文件中。

```
flashrl setup -a --fn bf16 -o ${CONFIG_PATH:-"$HOME/.flashrl_config.7b.yaml"}
```

#### 5、开始训练

脚本中添加以下环境变量：

```
# 打印详细日志，查看是否 patch 成功：
export FLASHRL_LOGGING_LEVEL=DEBUG
# 指定配置文件：
export FLASHRL_CONFIG=$HOME/.flashrl_config.7b.yaml
# 强制 lm-head 使用 bf16，减小精度损失：
export FLASHRL_LMHEAD_FP32=1
```

最后将 `tests/verl_examples/flash_rl/grpo_qwen25_7b_fsdp_int8_A2.sh` 和 `tests/verl_examples/configs/test_grpo_qwen25_7b_fsdp_int8_A2.sh` 两个文件复制到 verl，使用 verl 框架开启训练：

```
bash ./grpo_qwen25_7b_fsdp_int8_A2.sh
```

### 效果验证
由于量化开销较大，建议仅在模型规模较大（例如 14B 以上，最好是 32B 以上）且 COT 生成时间较长（使用 DAPO 训练而非 GSM8K 训练）的情况下使用量化 Rollout。

以下实验 bf16 和 int8 均开启[截断重要性采样](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)，lm-head 均使用 bf16 计算。

#### 精度： 
由于 GSM8K 数据集更容易收敛，用于验证精度:
<p align="left">
  <img src="../../sources/images/verl_npu/int8_accuracy.png" width="600"/>
</p>
bf16 收敛后的 reward 平均值为 0.8888，int8 收敛后的 reward 平均值为 0.8885，精度下降小于 0.1%。

bf16 每个 step 平均 e2e 耗时 48.9 s，int8 每个 step 平均 e2e 耗时 52.5 s，时间上升 6.9%。

#### 性能：
使用dapo-math-17k数据集，用于验证性能:
<p align="left">
  <img src="../../sources/images/verl_npu/int8_performance.png" width="600"/>
</p>
bf16 未收敛的 reward 平均值为 -0.8833，int8 未收敛后的 reward 平均值为 -0.8938，精度下降小于 1.2%

bf16 每个 step 平均 e2e 耗时 151.0 s，int8 每个 step 平均 e2e 耗时 116.2 s，时间降低 23.0%。