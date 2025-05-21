# MindSpeed RL 训练指标可视化
## 简介

- Weights & Biases（WandB）和 TensorBoard 都是机器学习领域广泛使用的实验跟踪与可视化工具。wandb功能更全面，可以展示丰富的数据类型，包括训练曲线，图片，视频，表格，html，matplotlib图像等，但是需要联网使用。而TensorBoard可以离线使用。
- MindSpeed RL支持训练指标的可视化：WandB和TensorBoard二选一使用。

## 使用示例
### TensorBoard使用
MindSpeed RL使用PyTorch原生的TensorBoard能力

#### 参数配置
**在训练yaml文件的rl_config字段中添加：**

```
# 开启tensorboard，若use_tensorboard和use_wandb同时为True，则tensorboard不生效
use_tensorboard: true   
```
#### 查看可视化指标
Step1. 开启`use_tensorboard`，在训练结束后，默认会在当前训练路径的`runs`目录中生成tensorboard Event格式日志
Step2. 终端输入命令：`tensorboard  --logdir=./runs  --host=$your_host_ip `
Step3. 浏览器打开`Step2`生成的url
tensorboard可视化训练指标效果示例：

![ScreenShot_20250320113451](../../sources/images/logging/logging_1.PNG)

### WandB使用
MindSpeed RL使用开源库WandB能力
#### 前置准备
1. 官网注册wandb账号，获取wandb login key
2. 确保训练环境能联网
3. 运行环境中设置wandb login key：export WANDB_API_KEY=$your_wandb_api_key
#### 参数配置
**在训练yaml文件的rl_config字段中添加：**
```
# 开启wandb
use_wandb: true            
wandb_project: "The_wandb_project_name"                   # 开启wandb时必填
wandb_exp_name: "The_wandb_experiment_name"               # 开启wandb时必填
wandb_save_dir: "Path_to_save_the_wandb_results_locally"  # 开启wandb时必填

```
#### 查看可视化指标
Step1. 开启`use_wandb`，配置`wandb_project`、`wandb_exp_name`、`wandb_save_dir`，开始训练。如果运行过程中wandb初始化失败则会默认切换到wandb离线模式，即只会生成wandb日志到设置的保存路径，不会同步到云端
wandb初始化成功会打印project url：

![ScreenShot_20250320155301](../../sources/images/logging/logging_4.PNG)


Step2. 浏览器打开`Step1`生成的wandb project url，查看训练指标。若当前环境不能联网，则可以将生成的日志拷贝到能联网的机器上，手动同步到云端：wandb sync $wandb日志保存路径

wandb可视化训练指标效果示例：

![ScreenShot_20250320113628](../../sources/images/logging/logging_2.PNG)

![ScreenShot_20250320113713](../../sources/images/logging/logging_3.PNG)
