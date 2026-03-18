由于DeepSeek-V3.2相关依赖的transformers、Megatron-Bridge适配PR尚未合入，因此需要以 **patch 补丁形式** 将关键适配点单独应用到现有版本中。

# 环境配套

| **组件**  | **配套版本** | **备注**                     |
| ----------------- | -------------------- | ------------------------------------ |
| python          | 3.11               |                                    |
| pytorch         | 2.8.0              |                                    |
| vllm            | v0.13.0            | commit 72506c98349                 |
| vllm-ascend     | releases/v0.13.0   | commit 0f812dcc58514               |
| verl            | main               | commit 0c06358d6b5624              |
| transformers    | 4.57.3       | commit 47b0e478f|
| Megatron        | main               | commit 1d462bd37dac21              |
| Megatron-Bridge |   main   | commit 7cabf71  |
| MindSpeed       | dev分支            |                                    |
| triton-ascend   | 需手动卸载         | 装了pip uninstall triton-ascend -y |

# 环境安装：

## 安装transformers

```
git clone https://github.com/huggingface/transformers.git 
cd transformers
git checkout 47b0e478f324b5
pip install -e .
cd ..
```

## 安装vllm

```
git clone https://github.com/vllm-project/vllm.git -b v0.13.0
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e . --no-build-isolation
cd ..
```

## 安装vllm_ascend

```
git clone https://github.com/vllm-project/vllm-ascend.git -b releases/v0.13.0
cd vllm-ascend
pip install -r requirements.txt
pip install -v -e .
cd ..
```

## 安装verl

```
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 0c06358d6b5624fe4e
pip install -r requirements-npu.txt
pip install -v -e .
cd ..
```

## 安装MindSpeed

```
git clone https://gitcode.com/Ascend/MindSpeed.git -b dev
cd MindSpeed
pip install -e .
cd ..
```

## 安装MindSpeedRL-patch

```
git clone https://gitcode.com/Ascend/MindSpeed-RL.git
cd transformers && git apply ../MindSpeed-RL/verl_npu/verl_npu/patch/transformers/47b0e478f/transformers.patch && pip install -e . && cd ..
cd vllm && git apply ../MindSpeed-RL/verl_npu/verl_npu/patch/vllm/72506c98349/common.patch && cd ..
cd vllm-ascend && git apply ../MindSpeed-RL/verl_npu/verl_npu/patch/vllm_ascend/0f812dcc58/sfa_v1.patch && cd..
cd verl && git apply ../MindSpeed-RL/verl_npu/verl_npu/patch/07056df535/dsa.patch && cd ..
cd MindSpeed && git apply ../MindSpeed/verl_npu/verl_npu/patch/mindspeed/07056df535 && cd ..
#需手动卸载triton
pip uninstall triton-ascend -y
```

## 安装Megatron-LM

```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 1d462bd37dac21
git apply ../MindSpeed-RL/verl_npu/verl_npu/patch/megatron-core/1d462bd37dac/rope_utils.patch
cp -r megatron ../verl/
cd ..
```

## 安装Megatron-Bridge

```
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout 7cabf71
git apply ../MindSpeed-RL/verl_npu/verl_npu/patch/megatron-bridge/7cabf71/model_bridge.patch
cp -r src/megatron/bridge ../verl/megatron
cd ..
```

# 模型运行

```
cd verl
cp ../MindSpeed-RL/tests/verl_examples/configs/test_grpo_deepseekv3.2exp_megatron_A3.sh ./
cp ../MindSpeed-RL/tests/verl_examples/grpo/grpo_deepseekv3.2exp_megatron_A3.sh.sh ./
```

复制后修改相应的配置，包括IP、网卡、节点数、模型路径与数据集路径等
`bash test_grpo_deepseekv3.2exp_megatron_A3.sh`