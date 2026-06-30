---
title: 模型权重的计算
published: 2026-06-30
description: 详解 Transformer Attention 机制和卷积层的参数量计算公式，涵盖 MSA、标准卷积及其常见变体。
tags: [AI, 模型参数量, Transformer, CNN]
category: tech blog
draft: false
---
对于 MLP 的权重不再做赘述，这里主要谈 Attention 和卷积层的权重计算。

## Attention

$$
Params = 3h^2
$$

其中 **h** 是 embedding size。

这是因为 Attention 机制

$$
\text{attention} = \text{softmax}\left(\frac{Q^TK}{\sqrt{d}}\right)V
$$

需要计算 3 个矩阵 $W^Q, W^K, W^V$，使得 $Q = W^QX, K = W^KX, V = W^VX$。这三个矩阵的大小都是 $h^2$。

> [!NOTE]
> 1. 一般而言，一个多头 Attention 后面还会接上一个 o_proj。所以，MSA 层参数 = attention 参数 + o_proj，也就是 $4h^2$。
> 2. 一个标准的 LLaMA Transformer block 是 RMSnorm → MSA（+x 残差） → RMSnorm → FFN（+x 残差）。
> 3. Flash Attention 不会改变参数量。

## 卷积层

$$
Params = K_h \times K_w \times C_{in} \times C_{out}
$$

对于一个输出通道的卷积核，它有三个步骤：

1. 给各个 channel 分配一个 $K_h \times K_w$ 大小的卷积核，执行卷积。
2. 将不同 channel 的卷积结果，按照位置相加，给出一层 output。
3. 所以，每个卷积核的权重是 $K_h \times K_w \times C_{in}$。

对于输出要有 $C_{out}$ 个的卷积层，自然要有对应数量的此类卷积核。

> [!NOTE]
> 1. 转置卷积和卷积参数相同。
> 2. 这是标准卷积，还有对卷积的改造，比如 **Depthwise Separable** 卷积可以减少卷积参数，具体看实现。
> 3. 一个标准的 ConvNeXt block 是 RMSnorm → depthwise conv 7×7（+x） → RMSnorm → FFN（+X）。

以上都忽略了偏置项，偏置是可以不设的。