
# 环境依赖

| 配置项| 版本信息|
| --- | --- |
| AI服务器 | Atlas 800T A3 64G*16 |
| 驱动固件 | 25.3.rc1 |
| Python | 3.10 |
|CANN  |  25.3.rc1 |
|torch | 2.7.1  |
| torch_npu |  2.7.1|
| transformers |  4.56.0 |
|vllm  | 0.11.0 |
| vllm-ascend | 0.11.0 |
|verl | 0.6.1-releasee |
|megatron-core | core_v0.12.1 |
|mindspeed |2.2.0_ core_r0.12.1 |
|mbridge | 0.13.1|

## 1.安装vllm、vllm-ascend、transformers、verl库

参考[verl_npu](../../README.MD)指导安装vllm、transformers、verl等依赖库

安装vllm-ascend：
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 650ce8ad1
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
cd ..
```
## 2.安装 MindSpeed 与 Megatron
```bash
# clone and setup MindSpeed (2.2.0_ core_r0.12.1)
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout a654440c
pip install -e .
cd ..

# clone and setup Megatron-LM (core_v0.12.1)
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout a845aa7
pip install -e .
# Megatron-LM 解决megatron.training问题
export PYTHONPATH=$PYTHONPATH:"Your Megatron-LM path"
cd ..
```

## 3.安装bridge
``` bash
# mbridge (0.13.1)
git clone https://github.com/ISEEKYAN/mbridge.git
cd mbridge
git checkout da21f04
pip install -e .
cd ..
```

## 4.安装插件
参考[verl_npu](../../README.MD)指导完成插件的安装，打上patch，若未打上执行下面的操作：
```bash
# 打开verl/__init__.py 找到`if is_npu_available:`，做如下添加
if is_npu_available:
	import verl_npu  # 添加上这一行
```

# gsm8k数据集预处理
## 参照verl处理gsm8k数据集
下载完gsm8k数据集后，修改verl/examples/data_preprocess/gsm8k.py文件中的内容：第50行`dataset = datasets.load_dataset(local_dataset_path, "main")`修改为本地gsm8k数据集路径，即包含train/test-00000-of-00001.parquet的父文件夹。参照预处理脚本，增加对应命令行参数，对gsm8k数据集进行预处理
