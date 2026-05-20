# 目录结构

## 概述

MindSpeed RL是基于昇腾生态的强化学习加速框架，提供端到端的RL训推解决方案。本文档详细说明仓库的目录结构与各模块功能。

## 目录结构总览

```text
MindSpeed-RL/
├── ci/                       # CI/CD流水线脚本
├── cli/                      # 命令行入口脚本
├── configs/                  # 训练配置文件
├── docs/                     # 项目文档
├── examples/                 # 训练示例脚本
├── mindspeed_rl/             # 核心RL训练框架
├── tests/                    # 测试用例
├── verl_npu/                 # verl昇腾NPU适配层
├── setup.py                  # 安装脚本
├── requirements.txt          # 依赖列表
├── README.md                 # 项目说明
├── LICENSE                   # 许可证
├── OWNERS                    # 仓库维护者
└── SECURITYNOTE.md           # 安全声明
```

## 各目录详细说明

### ci

CI/CD流水线相关脚本。

| 文件/目录 | 说明 |
| --------- | ---- |
| `access_control_test.py` | 访问控制测试脚本 |

### cli

命令行入口脚本，提供各训练算法的启动入口。

| 文件/目录 | 说明 |
| --------- | ---- |
| `train_dapo.py` | DAPO算法训练入口 |
| `train_dpo.py` | DPO算法训练入口 |
| `train_grpo.py` | GRPO算法训练入口 |
| `train_ppo.py` | PPO算法训练入口 |
| `preprocess_data.py` | 数据预处理入口 |
| `eplb_generate_map_ds.py` | EPLB映射数据集生成入口 |

### configs

训练配置文件目录，包含模型、数据集、算法等YAML配置。

| 文件/目录 | 说明 |
| --------- | ---- |
| `checkpoint/` | 检查点配置目录 |
| `checkpoint/model_cfg.json` | 模型检查点配置 |
| `datasets/` | 数据集配置目录 |
| `datasets/deepscaler.yaml` | DeepScaler数据集配置 |
| `datasets/math_17k.yaml` | MATH-17K数据集配置 |
| `datasets/orca_rlhf.yaml` | Orca-RLHF数据集配置 |
| `model/` | 模型配置目录 |
| `model/qwen25_7b.yaml` | Qwen2.5-7B模型配置 |
| `model/qwen25_32b.yaml` | Qwen2.5-32B模型配置 |
| `model/qwen3_8b.yaml` | Qwen3-8B模型配置 |
| `model/qwen3_32b.yaml` | Qwen3-32B模型配置 |
| `model/qwen3_30b_a3b.yaml` | Qwen3-30B-A3B模型配置 |
| `model/qwen3_235b_a22b.yaml` | Qwen3-235B-A22B模型配置 |
| `model/deepseekv3_671b.yaml` | DeepSeek-V3-671B模型配置 |
| `model/templates.json` | 模型模板配置 |
| `tools/` | 工具配置目录 |
| `tools/retool_config.yaml` | ReTool配置 |
| `tools/search_tool_config.yaml` | SearchTool配置 |
| `grpo_qwen25_7b_A3.yaml` | GRPO + Qwen2.5-7B A3配置 |
| `grpo_qwen25_32b_A2.yaml` | GRPO + Qwen2.5-32B A2配置 |
| `grpo_qwen25_32b_A3.yaml` | GRPO + Qwen2.5-32B A3配置 |
| `grpo_qwen3_8b_A3.yaml` | GRPO + Qwen3-8B A3配置 |
| `grpo_qwen3_235b_a22b_A2.yaml` | GRPO + Qwen3-235B-A22B A2配置 |
| `grpo_lora_qwen25_32b_A2.yaml` | GRPO LoRA + Qwen2.5-32B A2配置 |
| `grpo_deepseek_r1_671b_A2.yaml` | GRPO + DeepSeek-R1-671B A2配置 |
| `grpo_deepseek_r1_671b_A3.yaml` | GRPO + DeepSeek-R1-671B A3配置 |
| `grpo_deepseek_r1_671b_A3_eplb.yaml` | GRPO + DeepSeek-R1-671B A3 EPLB配置 |
| `dapo_qwen25_32b_A3.yaml` | DAPO + Qwen2.5-32B A3配置 |
| `dapo_qwen25_32b_A2_20k.yaml` | DAPO + Qwen2.5-32B A2 20K配置 |
| `dapo_qwen25_32b_A3_32k.yaml` | DAPO + Qwen2.5-32B A3 32K配置 |
| `dapo_qwen25_7b_A2_multi_turn.yaml` | DAPO + Qwen2.5-7B A2多轮配置 |
| `dapo_qwen3_30b_a3b_A3.yaml` | DAPO + Qwen3-30B-A3B A3配置 |
| `dapo_qwen3_32b_A3.yaml` | DAPO + Qwen3-32B A3配置 |
| `ppo_qwen25_32b_A3.yaml` | PPO + Qwen2.5-32B A3配置 |
| `dpo_qwen3_30b_a3b_A3.yaml` | DPO + Qwen3-30B-A3B A3配置 |

### docs

项目文档目录，包含算法说明、特性指南、解决方案等。

| 文件/目录 | 说明 |
| --------- | ---- |
| `zh/` | 中文文档目录 |
| `zh/algorithms/` | 算法说明文档 |
| `zh/algorithms/grpo.md` | GRPO算法使用指南 |
| `zh/algorithms/dapo.md` | DAPO算法使用指南 |
| `zh/algorithms/ppo.md` | PPO算法使用指南 |
| `zh/algorithms/dpo.md` | DPO算法使用指南 |
| `zh/features/` | 特性说明文档 |
| `zh/features/README.md` | 特性总览 |
| `zh/features/integrated_worker.md` | 训推共卡特性 |
| `zh/features/resharding.md` | 权重重切分特性 |
| `zh/features/remove_padding.md` | 填充移除特性 |
| `zh/features/context_parallel.md` | 长序列并行特性 |
| `zh/features/partial_rollout.md` | Partial Rollout特性 |
| `zh/features/expert_parallel.md` | 专家并行特性 |
| `zh/features/EPLB.md` | EPLB特性 |
| `zh/features/vpp.md` | VPP特性 |
| `zh/features/offload.md` | Offload特性 |
| `zh/features/recompute.md` | 重计算特性 |
| `zh/features/norm_recompute.md` | Norm重计算特性 |
| `zh/features/activation_function_recompute.md` | 激活函数重计算特性 |
| `zh/features/chunked_prefill.md` | Chunked Prefill特性 |
| `zh/features/acl_graph.md` | ACL Graph特性 |
| `zh/features/grpo_lora.md` | GRPO LoRA特性 |
| `zh/features/grpo_yaml.md` | GRPO YAML配置说明 |
| `zh/features/ppo_yaml.md` | PPO YAML配置说明 |
| `zh/features/multi_turn.md` | 多轮迭代训练特性 |
| `zh/features/data_module_design.md` | 数据调度模块设计 |
| `zh/features/task_queue.md` | 任务队列特性 |
| `zh/features/log_metrics.md` | 指标日志特性 |
| `zh/features/logging_wandb_tensorboard.md` | WandB/TensorBoard日志特性 |
| `zh/features/logging_swanlab.md` | SwanLab日志特性 |
| `zh/features/profiler.md` | 性能调优特性 |
| `zh/features/msprobe.md` | 精度分析特性 |
| `zh/features/deterministic_computation.md` | 确定性计算特性 |
| `zh/features/reuse_fp32_param.md` | FP32参数复用特性 |
| `zh/features/swap_attention.md` | Attention Swap特性 |
| `zh/features/swap_optimizer.md` | Optimizer Swap特性 |
| `zh/features/runtime_env.md` | 运行时环境配置 |
| `zh/features/vLLM_prefix_cache.md` | vLLM前缀缓存特性 |
| `zh/figures/` | 文档图片资源目录 |
| `zh/install_guide.md` | 安装指南 |
| `zh/solutions/` | 解决方案文档 |
| `zh/solutions/r1_zero_qwen25_7b.md` | R1-Zero + Qwen2.5-7B方案 |
| `zh/solutions/r1_zero_qwen25_32b.md` | R1-Zero + Qwen2.5-32B方案 |
| `zh/solutions/r1_zero_deepseek_671b.md` | R1-Zero + DeepSeek-671B方案 |
| `en/` | 英文文档目录 |
| `en/TODO` | 英文文档待办 |

### examples

训练示例脚本目录，提供各算法在不同模型上的Shell启动脚本。

| 文件/目录 | 说明 |
| --------- | ---- |
| `grpo/` | GRPO算法示例 |
| `grpo/grpo_trainer_qwen25_7b.sh` | GRPO + Qwen2.5-7B训练脚本 |
| `grpo/grpo_trainer_qwen25_32b.sh` | GRPO + Qwen2.5-32B训练脚本 |
| `grpo/grpo_trainer_qwen3_8b.sh` | GRPO + Qwen3-8B训练脚本 |
| `grpo/grpo_trainer_qwen3_235b_a22b.sh` | GRPO + Qwen3-235B-A22B训练脚本 |
| `grpo/grpo_trainer_deepseek_r1_671b.sh` | GRPO + DeepSeek-R1-671B训练脚本 |
| `dapo/` | DAPO算法示例 |
| `dapo/dapo_trainer_qwen25_32b.sh` | DAPO + Qwen2.5-32B训练脚本 |
| `dapo/dapo_trainer_qwen25_32b_20k.sh` | DAPO + Qwen2.5-32B 20K训练脚本 |
| `dapo/dapo_trainer_qwen25_32b_32k.sh` | DAPO + Qwen2.5-32B 32K训练脚本 |
| `dapo/dapo_trainer_qwen3_30b_a3b.sh` | DAPO + Qwen3-30B-A3B训练脚本 |
| `dapo/dapo_trainer_qwen3_32b.sh` | DAPO + Qwen3-32B训练脚本 |
| `ppo/` | PPO算法示例 |
| `ppo/ppo_trainer_qwen25_32b.sh` | PPO + Qwen2.5-32B训练脚本 |
| `dpo/` | DPO算法示例 |
| `dpo/dpo_trainer_qwen3_30b_a3b.sh` | DPO + Qwen3-30B-A3B训练脚本 |
| `eplb/` | EPLB示例 |
| `eplb/eplb.sh` | EPLB启动脚本 |
| `eplb/grpo_trainer_deepseek_r1_671b_eplb.sh` | GRPO + DeepSeek-R1-671B EPLB训练脚本 |
| `eplb/collect_json_file.sh` | JSON文件收集脚本 |
| `multi_turn/` | 多轮训练示例 |
| `multi_turn/dapo_trainer_qwen25_7b_multi_turn.sh` | DAPO + Qwen2.5-7B多轮训练脚本 |
| `data/` | 数据预处理示例 |
| `data/preprocess_data.sh` | 数据预处理脚本 |

### mindspeed_rl

核心RL训练框架，包含模型、训练器、数据集、工作器等核心模块。

| 文件/目录 | 说明 |
| --------- | ---- |
| `__init__.py` | 包初始化文件 |
| `config_cls/` | 配置类定义与校验 |
| `config_cls/base_config.py` | 基础配置类 |
| `config_cls/rl_config.py` | RL训练配置类 |
| `config_cls/megatron_config.py` | Megatron配置类 |
| `config_cls/generate_config.py` | 生成配置类 |
| `config_cls/data_handler_config.py` | 数据处理配置类 |
| `config_cls/mindstudio_config.py` | MindStudio配置类 |
| `config_cls/validate_config.py` | 配置校验 |
| `datasets/` | 数据集加载与预处理模块 |
| `datasets/base_dataset.py` | 基础数据集类 |
| `datasets/build_dataset.py` | 数据集构建 |
| `datasets/data_handler.py` | 数据处理器 |
| `datasets/dataloader.py` | 数据加载器 |
| `datasets/data_samplers.py` | 数据采样器 |
| `datasets/formatter.py` | 数据格式化 |
| `datasets/handler_utils.py` | 数据处理工具函数 |
| `datasets/indexed_dataset.py` | 索引数据集 |
| `datasets/prompt_dataset.py` | 提示数据集 |
| `datasets/multimodal_dataset.py` | 多模态数据集 |
| `datasets/mm_utils.py` | 多模态工具函数 |
| `datasets/preprocess_data.py` | 数据预处理 |
| `datasets/templates.py` | 数据模板 |
| `datasets/utils.py` | 数据集工具函数 |
| `models/` | 模型定义模块 |
| `models/actor.py` | Actor模型 |
| `models/actor_rollout_hybrid.py` | Actor-Rollout混合模型 |
| `models/critic.py` | Critic模型 |
| `models/reference.py` | Reference模型 |
| `models/reward.py` | Reward模型 |
| `models/math_dapo.py` | DAPO数学验证模型 |
| `models/math_verify.py` | 数学验证模型 |
| `models/rule_verifier.py` | 规则验证器 |
| `models/base/` | 模型基类 |
| `models/base/base_inference_engine.py` | 推理引擎基类 |
| `models/base/base_training_engine.py` | 训练引擎基类 |
| `models/loss/` | 损失函数模块 |
| `models/loss/base_loss_func.py` | 损失函数基类 |
| `models/loss/critic_loss_func.py` | Critic损失函数 |
| `models/loss/ppo_actor_loss_func.py` | PPO Actor损失函数 |
| `models/loss/grpo_actor_loss_func.py` | GRPO Actor损失函数 |
| `models/loss/dapo_actor_loss_func.py` | DAPO Actor损失函数 |
| `models/loss/reference_loss_func.py` | Reference损失函数 |
| `models/loss/reward_loss_func.py` | Reward损失函数 |
| `models/loss/logprob_computer.py` | 对数概率计算器 |
| `models/loss/loss_func_factory.py` | 损失函数工厂 |
| `models/loss/loss_register.py` | 损失函数注册 |
| `models/orm/` | ORM模型模块 |
| `models/orm/orm_model.py` | ORM模型 |
| `models/rollout/` | Rollout模块 |
| `models/rollout/vllm_engine.py` | vLLM推理引擎 |
| `models/rollout/vllm_adapter/` | vLLM适配器 |
| `models/rollout/vllm_adapter/engine_core.py` | vLLM引擎核心适配 |
| `models/rollout/vllm_adapter/fused_moe.py` | MoE融合适配 |
| `models/rollout/vllm_adapter/megatron_weight_loaders.py` | Megatron权重加载适配 |
| `models/rollout/vllm_adapter/vllm_parallel_state.py` | vLLM并行状态适配 |
| `models/rollout/vllm_adapter/patch/` | vLLM补丁 |
| `models/rollout/vllm_adapter/patch/qwen2_5_vl_reuse_vit_patch.py` | Qwen2.5-VL ViT复用补丁 |
| `models/rollout/vllm_adapter/patch/qwen2_5_vl_visionmlp_patch.py` | Qwen2.5-VL VisionMLP补丁 |
| `models/rollout/vllm_adapter/patch/rotary_embedding_patch.py` | 旋转嵌入补丁 |
| `tools/` | 工具集成模块 |
| `tools/base_tool.py` | 工具基类 |
| `tools/retool.py` | ReTool工具 |
| `tools/search_tool.py` | SearchTool工具 |
| `tools/utils/` | 工具辅助模块 |
| `tools/utils/execution_pool.py` | 执行池 |
| `tools/utils/schemas.py` | 工具Schema定义 |
| `tools/utils/tool_parser.py` | 工具解析器 |
| `tools/utils/tool_registry.py` | 工具注册表 |
| `tools/utils/tool_utils.py` | 工具工具函数 |
| `trainer/` | 训练器实现模块 |
| `trainer/base.py` | 训练器基类 |
| `trainer/grpo_trainer_hybrid.py` | GRPO混合训练器 |
| `trainer/dapo_trainer_hybrid.py` | DAPO混合训练器 |
| `trainer/ppo_trainer_hybrid.py` | PPO混合训练器 |
| `trainer/utils/` | 训练器辅助模块 |
| `trainer/utils/compute_utils.py` | 计算工具函数 |
| `trainer/utils/data_strategy.py` | 数据策略 |
| `trainer/utils/training.py` | 训练辅助函数 |
| `trainer/utils/parallel_state.py` | 并行状态管理 |
| `trainer/utils/transfer_dock.py` | 数据传输中转站 |
| `trainer/utils/mm_transfer_dock.py` | 多模态数据传输中转站 |
| `utils/` | 通用工具函数模块 |
| `utils/compute.py` | 计算工具函数 |
| `utils/context_parallel.py` | 上下文并行工具 |
| `utils/loggers.py` | 日志工具 |
| `utils/metrics.py` | 指标工具 |
| `utils/optimizer_module.py` | 优化器模块 |
| `utils/pad_process.py` | 填充处理 |
| `utils/remove_padding.py` | 填充移除 |
| `utils/seqLen_balancing.py` | 序列长度均衡 |
| `utils/tokenizer.py` | 分词器工具 |
| `utils/torch_functional.py` | PyTorch函数式工具 |
| `utils/utils.py` | 通用工具函数 |
| `utils/zmq_communication.py` | ZMQ通信工具 |
| `utils/math_eval_toolkit/` | 数学评估工具包 |
| `utils/math_eval_toolkit/grader.py` | 数学评分器 |
| `utils/math_eval_toolkit/parser.py` | 数学解析器 |
| `utils/transfer_queue/` | 传输队列模块 |
| `utils/transfer_queue/tq_client.py` | 传输队列客户端 |
| `utils/transfer_queue/tq_data.py` | 传输队列数据 |
| `utils/transfer_queue/tq_mgr.py` | 传输队列管理器 |
| `utils/transfer_queue/tq_utils.py` | 传输队列工具函数 |
| `workers/` | 分布式工作器模块 |
| `workers/base_worker.py` | 工作器基类 |
| `workers/actor_hybrid_worker.py` | Actor混合工作器 |
| `workers/critic_worker.py` | Critic工作器 |
| `workers/reference_worker.py` | Reference工作器 |
| `workers/reward_worker.py` | Reward工作器 |
| `workers/rule_reward.py` | 规则奖励工作器 |
| `workers/dynamic_sampling.py` | 动态采样工作器 |
| `workers/integrated_worker.py` | 集成工作器（训推共卡） |
| `workers/vit_worker.py` | ViT工作器 |
| `workers/eplb/` | EPLB模块 |
| `workers/eplb/eplb.py` | EPLB实现 |
| `workers/resharding/` | 权重重切分模块 |
| `workers/resharding/megatron_sharding_manager.py` | Megatron分片管理器 |
| `workers/resharding/megatron_off_loader.py` | Megatron卸载器 |
| `workers/resharding/memory_buffer.py` | 内存缓冲区 |
| `workers/resharding/vllm_weight_container.py` | vLLM权重容器 |
| `workers/resharding/weight_adaptor.py` | 权重适配器 |
| `workers/resharding/utils.py` | 重切分工具函数 |
| `workers/scheduler/` | 调度器模块 |
| `workers/scheduler/launcher.py` | 任务启动器 |
| `workers/scheduler/launcher_ms.py` | MindSpeed任务启动器 |
| `workers/scheduler/scheduler_ms.py` | MindSpeed调度器 |

### tests

测试用例目录，包含单元测试、系统测试和verl示例测试。

| 文件/目录 | 说明 |
| --------- | ---- |
| `conftest.py` | 测试配置 |
| `run_coverage.sh` | 覆盖率运行脚本 |
| `ut/` | 单元测试目录 |
| `ut/config/` | 配置模块单元测试 |
| `ut/config/test_config.py` | 配置类测试 |
| `ut/datasets/` | 数据集模块单元测试 |
| `ut/datasets/test_data_handler.py` | 数据处理器测试 |
| `ut/datasets/test_formatter.py` | 格式化测试 |
| `ut/datasets/test_handler_utils.py` | 数据处理工具测试 |
| `ut/datasets/test_indexed_dataset.py` | 索引数据集测试 |
| `ut/datasets/test_multimodal_dataset.py` | 多模态数据集测试 |
| `ut/datasets/test_prompt_dataset.py` | 提示数据集测试 |
| `ut/models/` | 模型模块单元测试 |
| `ut/models/test_actor.py` | Actor模型测试 |
| `ut/models/test_actor_rollout_hybrid.py` | Actor-Rollout混合模型测试 |
| `ut/models/test_critic.py` | Critic模型测试 |
| `ut/models/test_reference.py` | Reference模型测试 |
| `ut/models/test_reward.py` | Reward模型测试 |
| `ut/models/test_math_dapo.py` | DAPO数学模型测试 |
| `ut/models/test_verifier.py` | 验证器测试 |
| `ut/models/base/` | 模型基类测试 |
| `ut/models/base/test_base_training_engine.py` | 训练引擎基类测试 |
| `ut/models/loss/` | 损失函数单元测试 |
| `ut/models/loss/test_base_loss_func.py` | 损失函数基类测试 |
| `ut/models/loss/test_critic_loss_func.py` | Critic损失函数测试 |
| `ut/models/loss/test_ppo_loss_func.py` | PPO损失函数测试 |
| `ut/models/loss/test_grpo_loss_func.py` | GRPO损失函数测试 |
| `ut/models/loss/test_dapo_loss_func.py` | DAPO损失函数测试 |
| `ut/models/loss/test_dapo_actor_loss_func.py` | DAPO Actor损失函数测试 |
| `ut/models/loss/test_loss_register.py` | 损失函数注册测试 |
| `ut/utils/` | 工具模块单元测试 |
| `ut/utils/test_compute_utils.py` | 计算工具测试 |
| `ut/utils/test_tokenizer.py` | 分词器测试 |
| `ut/utils/test_pad_process.py` | 填充处理测试 |
| `ut/utils/test_torch_functional.py` | PyTorch函数式测试 |
| `ut/utils/test_utils.py` | 通用工具测试 |
| `ut/utils/test_logger.py` | 日志测试 |
| `ut/utils/test_grader.py` | 评分器测试 |
| `ut/utils/test_parser.py` | 解析器测试 |
| `ut/utils/test_parallel_state.py` | 并行状态测试 |
| `ut/utils/test_training.py` | 训练辅助测试 |
| `ut/utils/test_transfer_queue.py` | 传输队列测试 |
| `ut/utils/test_zmq_commnunication.py` | ZMQ通信测试 |
| `ut/workers/` | 工作器模块单元测试 |
| `ut/workers/test_base_worker.py` | 工作器基类测试 |
| `ut/workers/scheduler/` | 调度器测试 |
| `ut/workers/scheduler/test_launcher.py` | 启动器测试 |
| `ut/features/` | 特性单元测试 |
| `ut/features/reuse_fp32_param/` | FP32参数复用测试 |
| `ut/features/reuse_fp32_param/test_reuse_fp32_param.py` | FP32参数复用测试 |
| `st/` | 系统测试目录 |
| `st/st_run.sh` | 系统测试运行脚本 |
| `st/configs/` | 系统测试配置 |
| `st/configs/model/` | 系统测试模型配置 |
| `st/configs/model/qwen25_7b.yaml` | Qwen2.5-7B测试配置 |
| `st/configs/model/llama32-1b.yaml` | Llama3.2-1B测试配置 |
| `st/configs/test_grpo_trainer_qwen25_7b_integrated.yaml` | GRPO集成测试配置 |
| `st/configs/test_dapo_trainer_qwen25_7b_integrated.yaml` | DAPO集成测试配置 |
| `st/configs/test_ppo_trainer_qwen25_7b_integrated.yaml` | PPO集成测试配置 |
| `st/datasets/` | 数据集系统测试 |
| `st/datasets/test_preprocess_data.py` | 数据预处理测试 |
| `st/datasets/test_orca_rlhf.yaml` | Orca-RLHF测试配置 |
| `st/datasets/test_module_entry_preprocess_data.sh` | 数据预处理入口测试脚本 |
| `st/resharding/` | 重切分系统测试 |
| `st/resharding/test_resharding.py` | 重切分测试 |
| `st/resharding/test_module_entry_resharding_tp_expand.sh` | TP扩展重切分测试脚本 |
| `st/resharding/test_module_entry_resharding_tp_reduce.sh` | TP缩减重切分测试脚本 |
| `st/train_engine/` | 训练引擎系统测试 |
| `st/train_engine/test_module_entry_ray_grpo_qwen25_7b_tp4_integrated.sh` | GRPO集成测试脚本 |
| `st/train_engine/test_module_entry_ray_dapo_qwen25_7b_tp4_integrated.sh` | DAPO集成测试脚本 |
| `st/train_engine/test_module_entry_ray_ppo_qwen25_7b_tp4_integrated.sh` | PPO集成测试脚本 |
| `st/mindstudio/` | MindStudio系统测试 |
| `st/mindstudio/check_and_clean_mindstudio_output.py` | MindStudio输出清理脚本 |
| `test_tools/` | 测试工具 |
| `test_tools/dist_test.py` | 分布式测试工具 |
| `verl_examples/` | verl示例测试目录 |
| `verl_examples/configs/` | verl测试配置 |
| `verl_examples/configs/test_grpo_qwen25_7b_fsdp_int8_A2.sh` | GRPO + Qwen2.5-7B FSDP Int8测试 |
| `verl_examples/configs/test_grpo_qwen3_235b_A2.sh` | GRPO + Qwen3-235B测试 |
| `verl_examples/configs/test_dapo_qwen25_32b_fsdp2_A+X_4k.sh` | DAPO + Qwen2.5-32B FSDP2测试 |
| `verl_examples/configs/...` | 其他verl测试配置脚本 |
| `verl_examples/dapo/` | DAPO verl示例 |
| `verl_examples/dapo/dapo_qwen25_32b_fsdp2_A+X.sh` | DAPO + Qwen2.5-32B FSDP2脚本 |
| `verl_examples/dapo/...` | 其他DAPO verl示例脚本 |
| `verl_examples/grpo/` | GRPO verl示例 |
| `verl_examples/grpo/grpo_qwen3_235b_A2.sh` | GRPO + Qwen3-235B脚本 |
| `verl_examples/grpo/...` | 其他GRPO verl示例脚本 |
| `verl_examples/gspo/` | GSPO verl示例 |
| `verl_examples/gspo/gspo_qwen3_32b_A3.sh` | GSPO + Qwen3-32B脚本 |
| `verl_examples/flash_rl/` | FlashRL verl示例 |
| `verl_examples/flash_rl/grpo_qwen25_7b_fsdp_int8_A2.sh` | GRPO + Qwen2.5-7B FSDP Int8脚本 |
| `verl_examples/retool/` | ReTool verl示例 |
| `verl_examples/retool/run_qwen2_7b_dapo_npu.sh` | ReTool + Qwen2-7B DAPO NPU脚本 |
| `verl_examples/data_preprocess/` | 数据预处理verl示例 |
| `verl_examples/data_preprocess/deepscaler_json_to_parquet.py` | DeepScaler JSON转Parquet脚本 |

### verl_npu

verl昇腾NPU适配层，包含对verl/vLLM/Megatron/Transformers等组件的patch文件与插件。

| 文件/目录 | 说明 |
| --------- | ---- |
| `README.MD` | verl_npu说明文档 |
| `setup.py` | verl_npu安装脚本 |
| `requirements-npu-sgl.txt` | NPU SGL依赖列表 |
| `ci/` | CI/CD脚本 |
| `ci/access_control_test_verl.py` | verl访问控制测试 |
| `docs/` | verl_npu文档 |
| `docs/quick_start.md` | 快速入门 |
| `docs/mindspeed.md` | MindSpeed说明 |
| `docs/partial_rollout.md` | Partial Rollout说明 |
| `docs/quantized_rollout.md` | 量化Rollout说明 |
| `docs/deterministic_computation.md` | 确定性计算说明 |
| `docs/sglang_readme.md` | SGLang说明 |
| `docs/retool_zh.md` | ReTool中文说明 |
| `docs/model/` | 模型文档 |
| `docs/model/deepseek671b-v3.md` | DeepSeek-671B-V3模型说明 |
| `docs/model/deepseekv3.2.md` | DeepSeek-V3.2模型说明 |
| `docs/model/gptoss.md` | GPT-OSS模型说明 |
| `verl_npu/` | verl_npu核心代码 |
| `verl_npu/plugin.py` | 插件入口 |
| `verl_npu/patch/` | Patch文件目录 |
| `verl_npu/patch/mbridge/` | MBridge补丁 |
| `verl_npu/patch/megatron-bridge/` | Megatron Bridge补丁 |
| `verl_npu/patch/megatron-core/` | Megatron Core补丁 |
| `verl_npu/patch/mindspeed/` | MindSpeed补丁 |
| `verl_npu/patch/transformers/` | Transformers补丁 |
| `verl_npu/patch/verl/` | verl补丁 |
| `verl_npu/patch/vllm/` | vLLM补丁 |
| `verl_npu/patch/vllm_ascend/` | vLLM Ascend补丁 |
| `verl_npu/patch_summary/` | Patch摘要 |
| `verl_npu/patch_summary/patch_summary.py` | Patch摘要生成 |
| `verl_npu/patch_summary/patch_summary.yaml` | Patch摘要配置 |
| `verl_npu/utils/` | verl_npu工具 |
| `verl_npu/utils/pkg_version.py` | 包版本工具 |
| `verl_tests/` | verl测试目录 |
| `verl_tests/st/` | verl系统测试 |
| `verl_tests/st/st_run.sh` | 系统测试运行脚本 |
| `verl_tests/st/baseline_results/` | 基线结果 |
| `verl_tests/st/baseline_results/moonlight-16b-megatron.json` | Moonlight-16B Megatron基线 |
| `verl_tests/st/baseline_results/qwen3-8b-fsdp.json` | Qwen3-8B FSDP基线 |
| `verl_tests/st/train_engine/` | 训练引擎测试 |
| `verl_tests/st/train_engine/moonlight-16b-megatron.sh` | Moonlight-16B Megatron测试脚本 |
| `verl_tests/st/train_engine/qwen3-8b-fsdp.sh` | Qwen3-8B FSDP测试脚本 |
| `verl_tests/test_tools/` | verl测试工具 |
| `verl_tests/test_tools/conftest.py` | 测试配置 |
| `verl_tests/test_tools/acquire_json.py` | JSON获取工具 |
| `verl_tests/test_tools/test_ci_st.py` | CI系统测试 |

## 根目录文件说明

| 文件 | 说明 |
| ---- | ---- |
| `setup.py` | Python包安装脚本，用于pip安装mindspeed_rl |
| `requirements.txt` | Python依赖列表 |
| `README.md` | 项目说明文档 |
| `LICENSE` | Apache 2.0开源许可证 |
| `OWNERS` | 仓库维护者列表 |
| `SECURITYNOTE.md` | 安全声明文档 |
| `.gitignore` | Git忽略规则 |
