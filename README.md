  <p align="center"> <img src="./sources/images/logo.png" height="110px" width="500px"> </p>

# 简介

---



MindSpeed RL：基于昇腾生态的强化学习加速框架，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 生态合作伙伴提供端到端的RL训推解决方案，支持超大昇腾集群训推共卡/分离部署、多模型异步流水调度、训推异构切分通信等核心加速能力。

# 最新消息

---

🚀🚀🚀 2025.11 MindSpeed RL 已支持[agent rl多轮迭代训练](./docs/features/multi_turn.md) ！！！🚀🚀🚀

# 版本说明

---



MindSpeed RL依赖配套如下表：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">在研版本</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">在研版本</td>
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
    <td><a href="https://gitcode.com/ascend/pytorch#pytorch%E4%B8%8Epython%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%A1%A8">PyTorch与Python版本配套表</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.5</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
    <td rowspan="2">在研版本</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>

更多详情请参考[版本配套表](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/install_guide.md#%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%A1%A8)。


# 安装

---



MindSpeed RL具体的安装请参考[安装指南](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/install_guide.md#%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97)。

# 快速上手

---



  快速在昇腾训练设备上运行 MindSpeed RL可详见[快速入门](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md#grpo)。

# 使用指南

---



## 训练算法

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
      <td  rowspan="3"> GRPO</td>
      <td><a href="docs/algorithms/grpo.md">Doc</a></td>
      <td  rowspan="1">
        <a href="docs/solutions/r1_zero_qwen25_7b.md">Qwen2.5-7B</a> <br>
        <a href="docs/solutions/r1_zero_qwen25_32b.md">Qwen2.5-32B</a> <br>
      </td>
      <td> Released</td>
    </tr>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/rl">Doc</a></td>
      <td  rowspan="1">
        <a href="https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/rl">Qwen2.5VL-3B</a> <br>
        <a href="https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/rl">Qwen2.5VL-7B</a> <br>
        <a href="https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/rl">Qwen2.5VL-32B</a> <br>
      </td>
      <td> Released</td>
    </tr>
    <tr>
      <td><a href="docs/algorithms/grpo.md">Doc</a></td>
      <td  rowspan="1">
        <a href="examples/grpo/grpo_trainer_qwen3_8b.sh">Qwen3-8B</a> <br>
        <a href="examples/grpo/grpo_trainer_qwen3_235b_a22b.sh">Qwen3-235B-A22B</a> <br>
        <a href="docs/solutions/r1_zero_deepseek_671b.md"> DeepSeek-R1-671B </a> <br>
      </td>
      <td> Preview</td>
    <tr>
      <td> DAPO</td>
      <td><a href="docs/algorithms/dapo.md">Doc</a></td>
      <td  rowspan="1">
        <a href="examples/dapo/dapo_trainer_qwen25_32b.sh">Qwen2.5-32B</a> <br>
        <a href="examples/dapo/dapo_trainer_qwen3_30b_a3b.sh">Qwen3-30B-A3B</a> <br>
        <a href="examples/dapo/dapo_trainer_qwen3_32b.sh">Qwen3-32B</a> <br>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td> PPO</td>
      <td><a href="docs/algorithms/ppo.md">Doc</a></td>
      <td  rowspan="1">
        <a href="examples/ppo/ppo_trainer_qwen25_32b.sh">Qwen2.5-32B</a> <br>
      </td>
      <td> Preview</td>
    <tr>
      <td> DPO</td>
      <td><a href="docs/algorithms/dpo.md">Doc</a></td>
      <td  rowspan="1">
        <a href="examples/dpo/dpo_trainer_qwen3_30b_a3b.sh">Qwen3-30B-A3B</a> <br>
      </td>
      <td> Preview</td>
    </tr>
    </tr>
  </tbody>
</table>
注意："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本。

注意：使用Ray拉起任务的算法，如GRPO，环境变量需要在[runtime_env.yaml](./configs/envs/runtime_env.yaml)处配置。

## 核心特性

<table>
  <thead>
    <tr>
      <th>核心特性</th>
      <th>特性指南</th>
      <th>适用算法</th>
      <th>发布状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>训推共卡</td>
      <td><a href="docs/features/integrated_worker.md">Doc</a></td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
      </td>
      </td>
      <td> Released</td>
    </tr>
    <tr>
      <td>数据调度</td>
      <td><a href="docs/features/data_module_design.md">Doc</a></td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>权重重切分</td>
      <td><a href="docs/features/resharding.md">Doc</a></td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>填充移除</td>
      <td><a href="docs/features/remove_padding.md">Doc</a></td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>长序列并行</td>
      <td><a href="docs/features/context_parallel.md">Doc</a></td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
        DPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>Partial Rollout</td>
      <td><a href="docs/features/partial_rollout.md">Doc</a></td>
      <td  rowspan="1">
        GRPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
  </tbody>
</table>
注意："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本。

## 效率工具

<table>
  <thead>
    <tr>
      <th>工具特性</th>
      <th>特性指南</th>
      <th>适用算法</th>
      <th>发布状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>训练监控</td>
      <td>  <a href="docs/features/logging_wandb_tensorboard.md">Doc</a> </td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
        DPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>性能调优</td>
      <td>  <a href="docs/features/profiler.md">Doc</a> </td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
        DPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>精度分析</td>
      <td>  <a href="docs/features/msprobe.md">Doc</a> </td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
    <tr>
      <td>确定性计算</td>
      <td>  <a href="docs/features/deterministic_computation.md">Doc</a> </td>
      <td  rowspan="1">
        GRPO <br>
        DAPO <br>
        PPO <br>
        DPO <br>
      </td>
      </td>
      <td> Preview</td>
    </tr>
  </tbody>
</table>
注意："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本。

# 版本维护策略

----



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
| 2.2.0              | \            | 正式版本     |  商用      |   \  | \        | 2026/03/30 |
| 2.1.0              | \            | 正式版本     |  商用      |   \  | \        | 2025/12/30 |
| 2.0.0              | \            | 预览版本     |  预览      |   \  | \        | 2025/9/30 |

#  安全声明

---



[MindSpeed RL 安全声明](./SECURITYNOTE.md)



# 免责声明

---



## 致MindSpeed RL使用者

1. MindSpeed RL提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed RL平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed RL模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在gitcode提交issue，我们将及时审视并解决。
4. MindSpeed功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

## 致数据集所有者

如果您不希望您的数据集在MindSpeed RL中的模型被提及，或希望更新MindSpeed RL中的模型关于您的数据集的描述，请在gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed RL的理解和贡献。

## License声明

Ascend MindSpeed RL提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed RL根目录。

# 致谢

---



MindSpeed RL由华为公司的下列部门以及昇腾生态合作伙伴联合贡献 ：

华为公司：

- 计算产品线
- 2012实验室
- 公共开发部
- 全球技术服务部
- 华为云计算

生态合作伙伴:
- 工商银行大数据与人工智能实验室

感谢来自社区的每一个PR，共同打造业界领先的RL训推系统。
