---
title: 训练开销分析：显存占用与时间估算
published: 2026-06-30
description: 从显存占用和时间开销两个维度分析大模型训练成本，包含混合精度训练显存表、Chinchilla Scaling Laws 最优分配策略等核心内容。
tags: [AI, 训练开销, 显存, Scaling Laws, 分布式训练]
category: tech blog
draft: false
---

## 概述

> 模型显存开销取决于：模型参数量 × 精度，分布式训练方法也会影响。

参数量参考 [模型权重的计算](/posts/model-weights) 和 [模型参数量计算：推理与训练的内存占用](/posts/param-count)。

训练开销可以从两个维度来分析：显存占用和时间开销，二者的制约关系会影响我们选择何种分布式训练方法。

## 显存占用

显存占用即各参数 × 精度。

以 **混合精度 + AdamW + 全量训练** 举例：

| 项目 | 数据类型 | 每个参数字节 | 1.5B 模型占用 |
| --- | --- | --- | --- |
| 模型权重（前向） | FP16 | 2 | 3.0 GB |
| 梯度 | FP16 | 2 | 3.0 GB |
| 梯度副本 | FP32 | 4 | 6.0 GB |
| 优化器一阶矩 $m$ | FP32 | 4 | 6.0 GB |
| 优化器二阶矩 $v$ | FP32 | 4 | 6.0 GB |
| **基础占用合计** | | **16** | **24.0 GB** |
| 激活值 | FP16 | 2 | $\text{bytes} \times B \times S \times L \times (10H + 2 \times A \times S)$ |

激活值和训练时设置参数有关，可以通过梯度检查点技术来减少该部分开销。

> 分布式训练方法也会影响训练显存占用。比如说 Deepseed ZeRO v3 会保持总显存占用，DDP 方法则会在每个卡上加载所需显存量，所以 n 卡则显存占用 × n。
> 
> 以上论述的都是**单卡**的情况。

## 时间开销

首先计算计算量：

$$
\text{计算量} \approx 6P \times D \ \text{FLOPS}
$$

其中 P 是参数量，D 是数据量（token 量），这里的 6 是经验参数，源自前向传播 2 FLOPS，反向传播 4 FLOPS。

$$
\text{计算时间} \approx \frac{\text{计算量}}{\text{算力}}
$$

算力 = 卡数 × 单卡性能 × 通信卡销折扣。通信开销折扣取决于使用的多卡操作方法，比如 DP、DDP、Deepseed ZeRO。

> 举例：假设我们已经确定了要**训练的模型**和已有的**显卡**，然后我们的训练有 **DDL**，我们就可以根据这两者选择我们的**分布式训练方法**和**训练数据量**，以求在 DDL 前训练完。

### Chinchilla Scaling Laws

**在固定的计算预算（Compute Budget）下，模型参数规模（N）和训练数据量（D）应如何分配，才能使模型最终性能达到最优。**

计算量表示为：

$$
C \approx c \times P \times D
$$

这里 c 经验上是 6。Chinchilla Scaling Laws 提出：

$$
L(P,D) = E + \frac{A}{P^\alpha} + \frac{B}{D^\beta}
$$

- $L(P,D)$：是预训练损失
- $E$：irreducible loss，代表数据本身的信息熵，是损失的理论下限
- $A, B, \alpha, \beta$：经验参数，实验拟合

所以求最优模型大小和数据量变成了一个优化问题：

$$
\begin{align}
P^*, D^* = &\arg \min_{P,D} L(P,D) = E + \frac{A}{P^\alpha} + \frac{B}{D^\beta} \\
&\text{s.t.} \quad c \times P \times D = C
\end{align}
$$

DeepMind 实验给出一个最优情况：70B 模型时 $D = 20P$，又发现 $\alpha \approx \beta$，意味着二者同步增长保持最优。所以最优情况就是 $D = 20P$。

> [!NOTE]
> 但是实践上，受限于各种因素，我们不一定会遵循以上规律。比如 DeepSeek Flash 系列基本遵循了，但大部分时候是模型规模不能提升，却仍然猛猛喂数据——因为数据多。

一般情况下，模型规模固定时，可以独立增大数据量来提升性能（即所谓的"过训练"策略）。
