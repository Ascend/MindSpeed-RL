# 环境依赖

## 
| MindSpeed RL版本 | PyTorch版本 | torch_npu版本 | CANN版本  | Python版本 |
| ---------------- | ------------ |-----------| ---------- | ---------- |
| master（主线）   | 2.7.1     | 2.7.1       | 8.3.RC1 | Python3.10 |

## 激活cann
```bash
# 修改为自己的cann路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 1、安装 vllm 和 vllm-ascend
```bash
# vllm     0.10.1
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 38217877aa70041c0115ee367b75197af9cbc5ad
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend   0.10.0 
# 安装vllm_ascend时，注释requirements.txt和pyproject.toml中torch、torchvision和torch_npu的安装
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 1de16ead8eecfec8903ec1b330b27a4fa2593c35
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
pip install torchvision==0.22.1 --extra-index https://download.pytorch.org/whl/cpu/
cd ..

# 源码安装transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8365f70e925
pip install -e .
```

## 2、安装 MindSpeed 与 Megatron
```bash
# MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 1cdd0abd75e40936ad31721c092f57c695dd72c4
pip install -e .
cd ..

# Megatron
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1
```

## 3、安装 verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 796871d7d092f7cbc6a64e7f4a3796f7a2217f5e
pip install -e .
cd ..
```

## 4、安装插件
```bash
# 请确保 vllm 已正确安装并且之后不会做覆盖
git clone -b 2.3.0 https://gitcode.com/Ascend/MindSpeed-RL.git
cd MindSpeed-RL/verl_npu
pip install -v -e .
cd ../..
```

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```

**注意**：请在安装完插件后做如下检查，确保插件安装成功

~~~bash
# 使用verl拉起训练时检查是否有如下输出：
================================ NPU Patch Summary ==================================

================ verl Patch Summary ================

Patch File1: verl.workers.sharding_manager.hybrid_tp_config.py
  (1) Patch class: verl.workers.sharding_manager.hybrid_tp_config.hybrid_tp_config
       Class Changes:
          - added         module_attr         Dict
          - added         module_attr         DictConfig
          - added         module_attr         HybridTPConfig
          - added         module_attr         List
          - added         module_attr         Optional
          - added         module_attr         dataclass

Patch File2: verl.workers.megatron_workers.py
  (1) Patch class: verl.workers.megatron_workers.ActorRolloutRefWorker
       Class Changes:
          - replaced         method         compute_log_prob
          - replaced         method         update_actor

Patch File3: verl.utils.seqlen_balancing.py

Patch File4: verl.workers.rollout.vllm_rollout.vllm_rollout_spmd.py
  (1) Patch class: verl.workers.rollout.vllm_rollout.vllm_rollout_spmd.vLLMRollout
       Class Changes:
          - replaced         method         __init__
          - replaced         method         _init_dp_env

Patch File5: recipe.dapo.dapo_ray_trainer.py
  (1) Patch class: recipe.dapo.dapo_ray_trainer.RayDAPOTrainer

============ verl Patch Summary End ==============

================ transformers Patch Summary ================

Patch File1: src.transformers.models.qwen2.modeling_qwen2.py
   Module Changes:
          - added         method         fused_apply_rotary_pos_emb
  (1) Patch class: src.transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
       Class Changes:
          - replaced         method         forward
  (2) Patch class: src.transformers.models.qwen2.modeling_qwen2.Qwen2MLP
       Class Changes:
          - replaced         method         forward

Patch File2: src.transformers.models.qwen3_moe.modeling_qwen3_moe.py
   Module Changes:
          - added         method         apply_rotary_pos_emb
  (1) Patch class: src.transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
       Class Changes:
          - replaced         method         __init__
  (2) Patch class: src.transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm
       Class Changes:
          - replaced         method         forward
  (3) Patch class: src.transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeMLP
       Class Changes:
          - replaced         method         forward

Patch File3: src.transformers.integrations.npu_flash_attention.py
   Module Changes:
          - added         method         unpad_input
          - added         method         _prepare_from_posids
  (1) Patch class: src.transformers.integrations.npu_flash_attention.IndexFirstAxis
  (2) Patch class: src.transformers.integrations.npu_flash_attention.IndexPutFirstAxis
  (3) Patch class: src.transformers.integrations.npu_flash_attention.pad_input

============ transformers Patch Summary End ==============

============================= NPU Patch Summary End==================================
~~~

若没有，则执行下面的操作：

~~~bash
# 打开verl/__init__.py 找到`if is_npu_available:`，做如下添加
if is_npu_available:
	import verl_npu  # 添加上这一行
~~~

## 5、内存管理优化（可选）

安装 jemalloc 以优化内存管理

可通过源码编译安装，前往官方仓库获取最新稳定版本

安装步骤：
``` python
    tar -xvf jemalloc-{version}.tar.bz2
    cd jemalloc-{version}
    ./configure --prefix=/usr/local
    make
    make install
```

安装完成后设置环境变量(假设安装路径为 `/usr/local/lib/libjemalloc.so.2`): 
``` python
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

## 6、虚拟内存动态开启关闭
在强化学习训练场景中，训练过程易产生大量内存碎片；同时推理阶段启用的 sleepmode 机制与虚拟内存存在兼容性冲突。为此，本特性支持虚拟内存的动态开启与关闭，可有效减少强化学习训练场景下的内存碎片问题，降低训练过程中发生 **内存溢出（OOM）** 的风险。
插件安装完成后，可通过以下方式启用该特性：
```bash
export EXPANDABLE_SEGMENTS=True
```

# 启动训练

安装成功后，将 `MindSpeed-RL/tests/verl_examples` 下提供的参考配置脚本放入 verl 目录下，具体为：

`configs` 目录提供具体模型及算法配置

`dapo`及`grpo`目录提供与 `configs` 对应的执行脚本，运行时配置好该脚本中的 `DEFAULT_SH` 即可拉起

# verl_npu开发

## verl_npu功能特性

verl_npu提供下述功能特性来实现极简易用的NPU集成：

1. **Patch 注入** - 使用git原生patch程序对Patch文件注入

2. **Patch Summary** - 对Patch注入结果进行概述

3. **Patch 版本管理** - 对当前依赖库多个版本进行统一

4. **Patch 结果探测** - 探测Patch注入结果

## 开发指南

### 快速开始

如果您需要为新的NPU硬件或新的verl版本开发插件，请参考：

- **[快速开始指南](docs/quick_start.md)** - 快速了解框架概览

### 验证开发结果

verl_npu开发完成后，按照上述安装步骤重新安装插件.