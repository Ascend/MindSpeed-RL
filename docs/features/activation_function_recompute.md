### 激活函数重计算

## 背景介绍

在当前大模型训练场景中，混合精度训练已成为标准实践，其中涉及计算权重与状态权重的持续存储。然而，这两类权重的生命周期并不重叠，这意味着它们可以共享内存空间，而非各自独立占用。通过数值变换技巧，可以消除这一冗余，实现资源的有效利用。

在大规模模型训练过程中，前向传播会产生海量中间激活值。由于这些激活值需要在后续的反向传播过程中参与梯度计算，系统必须在整个计算周期内持续保存这些大型张量，直至对应的梯度计算完成为止。这种存储需求导致显存占用急剧上升，成为制约模型规模扩展的主要瓶颈之一。为应对这一挑战，业界提出了重计算技术，其核心思想是在前向传播过程中选择性丢弃部分中间结果，当反向传播需要时再重新执行对应的前向计算。这种"以时间换空间"的策略有效缓解了显存压力，为大模型训练提供了可行的解决方案。

然而，现有主流训练框架普遍采用将重计算与反向传播紧密耦合的设计架构。这种强耦合关系导致系统在前向传播阶段仍需保存大量中间激活值，限制了重计算技术的优化潜力。

例如，在模型中存在以下流程：

* 前向：gelu激活函数模块->后续模块A
* 反向：后续模块A的反向（需要gelu输出的激活值）->gelu反向（与重计算绑定）

gelu激活函数会产生大量的数据，但本身计算量很小。此时进行激活函数的重计算可以在性能劣化极少的代价下，减少内存占用。 但在现有重计算框架下，如果对gelu激活函数模块做重计算，并不能节省gelu函数的输出。这是因为在反向时，模块A所需要的gelu输出的激活值，会早于gelu激活函数模块的重计算流程，所以前向必须保留激活函数的输出，导致激活函数的输出并不能节省下来。

## 方案介绍

本方案重新设计并实现了一套重计算框架，以轻微的时间损耗为代价，显著降低了显存占用，同时保证了重计算策略的高度灵活性——重计算操作可以被灵活地插入到反向计算开始前的任意位置。

新框架的流程如下：

反向（新框架）：gelu函数重计算->后续模块A的反向

在这一流程中，由于gelu函数的输出是通过重计算在模块A的反向计算之前即时生成的，因此在前向传播阶段无需保留gelu函数的输出值，从而实现了显存的节约。

相比之下，传统重计算方案虽然能够减少部分计算开销，但由于其紧密耦合的设计，导致在前向过程中仍需保留大量中间激活值，显存优化效果有限。

而本方案设计了一种传入模块函数进行重计算的机制，在合适的时机，丢弃重计算模块输出的物理存储，保留逻辑视图。在反向时，在合适的时机，利用register_hook插入重计算流程。利用传入的函数重新进行计算，得到结果。

##### 图1 重计算与反向绑定

![](https://img-transfer.gitcode.com/?p=https%3A%2F%2Fgitee.com%2Fascend%2FMindSpeed%2Fraw%2Fmaster%2Fsources%2Fimages%2Factivation_function_a.png&projectId=wangshuyang31/MindSpeed-RL&pageUrl=https%3A%2F%2Fgitcode.com%2Fwangshuyang31%2FMindSpeed-RL%2Fdiscussions%2F1)

比如gelu在MLP中的位置如图1所示。反向计算需要前向产生的a、b、c、d。其中b和c的shape为(batch, seq , 4hidden_size)，gelu为激活函数，其计算较少，故可将tensor c释放掉，反向在4h->h反向前重新计算。

##### 图2 灵活插入重计算

![](https://img-transfer.gitcode.com/?p=https%3A%2F%2Fgitee.com%2Fascend%2FMindSpeed%2Fraw%2Fmaster%2Fsources%2Fimages%2Factivation_function_b.png&projectId=wangshuyang31/MindSpeed-RL&pageUrl=https%3A%2F%2Fgitcode.com%2Fwangshuyang31%2FMindSpeed-RL%2Fdiscussions%2F1)

在前向4h->h计算完毕后，将c释放，保留逻辑视图。在4h->h grad前，需要将c计算回来。如图2所示，这里使用给d打tensor_hook的方式来进行重计算的插入。

## 使用介绍

主要用于训练场景，用户内存不足或要节省内存时，可以开启激活函数重计算，节省激活函数的输出激活值。

脚本中添加：`--recompute-activation-function` 可开启激活函数重计算。

添加：`--recompute-activation-function-num-layers ${num}` 可指定激活函数重计算的层数。

#### 说明

激活函数重计算可以与全重计算同时开启：

* 同时开启时，仅支持 `--recompute-method 为 block`
* 同时开启时，会按照指定的全重计算和激活函数重计算的层数做各自类型的重计算，即不会有一层既做全重计算又做激活函数重计算。

执行优先级是先计算全重计算层，后计算激活函数重计算层。在流水线并行未开启的情况下，全重计算层数和激活函数重计算层数之和应该等于总层数。

* 暂不兼容自适应重计算特性。

## 使用效果

启用激活函数重计算后，激活函数的输出激活值不用再保存，内存占用减少。同时由于激活函数计算量很小，训练性能只会略微下降。

根据模型配置不同，激活函数重计算收益也会发生改变。

## 扩展使用

本特性引入的 CheckpointWithoutOutput 类可以自定义对任何模块进行重计算，并且在合适的时机进行重计算恢复。

此处提供一个示例，可以灵活使用 CheckpointWithoutOutput 来对自定义的函数进行重计算：

（注意：如要使用register_hook，需要确保张量有梯度）

```python
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput


class Custom_module(torch.nn.Module):
    def __init__(self):
        ......

    def forward(self, input):
        self.activation_checkpoint_manager = CheckpointWithoutOutput()
        function_output = self.activation_checkpoint_manager.checkpoint(self.custom_function, False, function_input1, function_input2, ...)
        ...(after used output)
        self.activation_checkpoint_manager.discard_output()
        if module_output.requires_grad:
            module_output.register_hook(self.activation_checkpoint_manager.recompute)

        return module_output
```
