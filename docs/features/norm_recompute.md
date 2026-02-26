# Norm重计算

## 背景与挑战

大模型训练过程中，往往会面临的显存不足的问题。

## 解决方案

类似于激活函数重计算，本特性支持了Norm层的重计算，运用激活函数重计算特性中的checkpoint机制，对Norm层进行重计算处理。
具体细节可参见文献[Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism](https://www.usenix.org/conference/atc24/presentation/yuan)。

## 使用场景

主要用于训练场景，用户内存不足或要进一步节省内存时。

## 使用方法

需在训练脚本中加入以下参数配置。

`--recompute-norm  # 开启Norm重计算`
`--recompute-norm-num-layers ${num}   # num表示Norm重计算的层数`

#### 说明
* Norm重计算特性仅支持mcore分支，不支持legacy分支，即仅支持在开启`--use-mcore-models`时，通过`--recompute-norm`使能。
* Norm重计算兼容激活函数重计算、全重计算同时开启：
    * 同时开启时，仅支持--recompute-method设置为block。
    * 同时开启时，将按照指定的全重计算和Norm重计算的层数做各自类型的重计算，即不会有一层既做全重计算又做Norm重计算。

* 执行优先级是先计算全重计算层，后Norm重计算层。

## 使用效果
开启后可节省RMSNorm/LayerNorm层的输出激活内存，并且由于Norm计算速度较快，重计算后对整体性能影响较小。对于开启TP及SP的场景，由于该激活内存在TP域内已进行切分，开启后效果不明显，对于未使用TP及SP的模型可考虑使用。