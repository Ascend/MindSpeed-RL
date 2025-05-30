  <p align="center"> <img src="./sources/images/logo.png" height="110px" width="500px"> </p>



MindSpeed RL是基于昇腾生态的强化学习加速框架，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 生态合作伙伴提供端到端的RL训推解决方案，支持超大昇腾集群训推共卡/分离部署、多模型异步流水调度、训推异构切分通信等核心加速能力



## 安装指南

本版本为**预览非正式发布**版本， 依赖配套如下表，安装步骤参考[安装指南](./docs/install_guide.md)

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">25.0.RC1</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">8.1.RC1</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>torch</td>
    <td rowspan="2">2.5.1</td>
  </tr>
  <tr>
    <td>torch_npu</td>
  </tr>
  <tr>
    <td>apex</td>
    <td rowspan="1">0.1</td>
  </tr>
  <tr>
    <td>ray</td>
    <td>2.42.1</td>
  </tr>
  <tr>
    <td>vllm</td>
    <td>0.7.3</td>
  </tr>
</table>

## 使用指南

### 训练算法

<table>
  <thead>
    <tr>
      <th>训练算法</th>
      <th>训练指南</th>
      <th>支持模型</th>
      <th>发布状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> GRPO</td>
      <td><a href="docs/algorithms/grpo.md">Doc</a></td>
      <td  rowspan="1">
        Qwen2.5-7B <br>
        Qwen2.5-32B <br>
      </td>
      <td> Preview</td>
    </tr>
  </tbody>
</table>
注："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本 

注：使用Ray拉起任务的算法，如GRPO，环境变量需要在[runtime_env.yaml](./configs/envs/runtime_env.yaml)处配置

### 核心特性

<table>
  <thead>
    <tr>
      <th>核心特性</th>
      <th>特性指南</th>
      <th>适用算法</th>
      <th>适用模型</th>
      <th>发布状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>训推共卡</td>
      <td><a href="docs/features/integrated_worker.md">Doc</a></td>
      <td  rowspan="1">GRPO</td>
      <td  rowspan="1">
        Qwen2.5-7B <br>
        Qwen2.5-32B <br>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>数据调度</td>
      <td><a href="docs/features/data_module_design.md">Doc</a></td>
      <td  rowspan="1">GRPO</td>
      <td  rowspan="1">
        Qwen2.5-7B <br>
        Qwen2.5-32B <br>
      </td>
      <td> Preview</td>
    </tr>
  </tbody>

</table>
注："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本 

### 效率工具

<table>
  <thead>
    <tr>
      <th>工具特性</th>
      <th>特性指南</th>
      <th>适用算法</th>
      <th>适用模型</th>
      <th>发布状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>训练监控</td>
      <td>  <a href="docs/features/logging_wandb_tensorboard.md">Doc</a> </td>
      <td  rowspan="1">GRPO</td>
      <td  rowspan="1">
        Qwen2.5-7B <br>
        Qwen2.5-32B <br>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>性能调优</td>
      <td>  <a href="docs/features/profiler.md">Doc</a> </td>
      <td  rowspan="1">GRPO</td>
      <td  rowspan="1">
        Qwen2.5-7B <br>
        Qwen2.5-32B <br>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>数据采集</td>
      <td>  <a href="docs/features/msprobe.md">Doc</a> </td>
      <td  rowspan="1">GRPO</td>
      <td  rowspan="1">
        Qwen2.5-7B <br>
        Qwen2.5-32B <br>
      </td>
      <td> Preview</td>
    </tr>
  </tbody>
</table>
注："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本 

## 版本维护策略

MindSpeed RL版本有以下五个维护阶段：

| **状态**            | **时间**  | **说明**                                                                  |
| ------------------- | --------- |-------------------------------------------------------------------------|
| 计划                | 1—3 个月  | 计划特性                                                                    |
| 开发                | 3 个月    | 开发特性                                                                    |
| 维护                | 6-12 个月 | 合入所有已解决的问题并发布版本，针对不同的MindSpeed RL版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月  | 合入所有已解决的问题，无专职维护人员，无版本发布                                                |
| 生命周期终止（EOL） | N/A       | 分支不再接受任何修改                                                              |


MindSpeed RL已发布版本维护策略：

| **MindSpeed RL版本** | **对应标签** | **维护策略** | **当前状态** | **发布时间**  | **后续状态** | **EOL日期** |
|--------------------| ------------ |----------|----------|-----------|----------|-----------|
| 2.0.0              | \            | 预览版本     |  预览      |   \  | \        | 2025/9/30 |


## 安全声明
详细[安全声明](./SECURITYNOTE.md)


## 致谢

MindSpeed RL由华为公司的下列部门以及昇腾生态合作伙伴联合贡献 ：

华为公司：

- 计算产品线
- 2012实验室 
- 公共开发部
- 全球技术服务部
- 华为云计算

感谢来自社区的每一个PR，共同打造业界领先的RL训推系统


## 免责声明

### 致MindSpeed RL使用者
1. MindSpeed RL提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed RL平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed RL模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

### 致数据集所有者
如果您不希望您的数据集在MindSpeed RL中的模型被提及，或希望更新MindSpeed RL中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed RL的理解和贡献。
