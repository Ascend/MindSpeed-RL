# Sglang安装指导说明


## 1. 环境依赖
| MindSpeed RL版本 | PyTorch版本 | torch_npu版本 | triton-ascend版本 | CANN版本 | Python版本 |
| ---------------- | ----------- | ------------- | ----------------- | -------- | ---------- |
| master（主线）   | 2.7.1       | 2.7.1         | 3.2.0             | 8.5.0  | Python3.11 |

## 2. 安装 sglang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout 2fb31605985d650e969ac27fd3ec026a9de9c621
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[srt_npu]"
```

## 3. 激活CANN包并安装torch相关包

**注意**：[PyTorch框架和torch_npu插件安装教程](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)；可从[PyTorch-Ascend官方代码仓](https://gitcode.com/Ascend/pytorch/releases)获取PyTorch各个版本对应的torch_npu的whl包。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

pip install torch-2.7.1-cp311-cp311-*
pip install torch_npu-2.7.1-cp311-cp311-*
pip install -i https://test.pypi.org/simple/ --trusted-host=test-files.pythonhosted.org --trusted-host=test.pypi.org triton_ascend==3.2.0.dev20251226
```


## 4. 安装 sgl-kernel-npu

```bash
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu
git checkout 1221875d0c4095f8628d52811d21ebb1d0660e91
# 请确保CANN的路径为默认路径/usr/local/Ascend/下，如果不是，则需要设置软链接。
bash build.sh
pip install output/torch_memory_saver*.whl
pip install output/sgl_kernel_npu*.whl

# build.sh脚本中默认是构建A3机器上的DeepEP-Ascend包。如果是A2机器，请单独构建，方法如下：
bash build.sh -a deepep2

pip install output/deep_ep*.whl
# 设置deep_ep_cpp*.so的软链接
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -

# 确认是否可以成功导入
python -c "import deep_ep; print(deep_ep.__path__)"
python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
cd ..
```

## 5. 安装 verl

**注意**：安装前需要将 `MindSpeed-RL/verl_npu` 下提供的 **[requirements-npu-sgl.txt](../requirements-npu-sgl.txt)** 移至verl文件夹下。

```bash
# verl v0.6.1版本
git clone https://github.com/volcengine/verl.git
cd verl
git checkout d62da4950573d7a4b7ef2362337952e7ab59e78d
pip install --no-deps -e .
pip install -r requirements-npu-sgl.txt
pip uninstall timm
```


## 6. 安装插件

```bash
# 请确保 sglang 已正确安装并且之后不会做覆盖
git clone https://gitcode.com/Ascend/MindSpeed-RL.git
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

 Patch File2: verl.workers.rollout.vllm_rollout.vllm_rollout_spmd.py
   (1) Patch class: verl.workers.rollout.vllm_rollout.vllm_rollout_spmd.vLLMRollout
        Class Changes:
           - replaced         method         __init__
           - replaced         method         _init_dp_env

 ============ verl Patch Summary End ==============

 ============================= NPU Patch Summary End==================================
~~~

若没有，则执行下面的操作：

~~~bash
# 打开verl/__init__.py 找到`if is_npu_available:`，做如下添加
if is_npu_available:
	import verl_npu  # 添加上这一行
~~~

## 7. 启动训练

这里以Qwen2.5-32B为例，基于GRPO与规则奖励，使用deepscaler数据集。

### 7.1 数据集使用（可选）

#### 下载数据集：

deepscaler数据集下载链接：[huggingface: DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/blob/main/deepscaler.json)。

#### 数据集预处理：

在使用verl框架进行训练时，需要将该数据集从json文件转换为parquet文件。提供转换脚本，在数据文件格式转换的同时给prompt增加了激发模型思考的模板：[verl: deepscaler_json_to_parquet.py](../../tests/verl_examples/data_preprocess/deepscaler_json_to_parquet.py)。**需要修改其中的`input_file_path、output_file_path`为数据集实际读取和保存路径。**

将`deepscaler_json_to_parquet.py`文件移至verl项目中，比如 `verl/examples/data_preprocess/` 文件夹下，执行命令示例如下：
```bash
python examples/data_preprocess/deepscaler_json_to_parquet.py
```

### 7.2 训练脚本

安装成功后，将 `MindSpeed-RL/tests/verl_examples` 下提供的参考配置脚本放入 verl 目录下，具体为：

`configs` 目录提供具体模型及算法配置，参考脚本地址为：**[test_grpo_qwen2.5_32b_sglang_A2_8k.sh](../../tests/verl_examples/configs/test_grpo_qwen2.5_32b_sglang_A2_8k.sh)**

**注意**：该脚本中 **数据集、权重和日志的路径** 等配置需按具体使用情况修改，此处默认日志保存在`verl/logs/`文件夹下。

`dapo`及`grpo`目录提供与 `configs` 对应的执行脚本，运行时需要配置好该脚本中的 `DEFAULT_SH` 路径，参考脚本地址为：**[grpo_qwen2.5_32b_sglang_A2.sh](../../tests/verl_examples/grpo/grpo_qwen2.5_32b_sglang_A2.sh)**

**注意**：该脚本中 **用例脚本地址、IP和网卡** 等配置需按具体使用情况修改。