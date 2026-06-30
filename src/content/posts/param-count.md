---
title: 模型参数量计算：推理与训练的内存占用
published: 2026-06-30
description: 详解模型推理和训练时的参数量与显存占用计算，涵盖 KV Cache、混合精度训练、AdamW 优化器及 LoRA 微调的内存分析。
tags: [AI, 模型参数量, 显存, KV Cache, LoRA]
category: tech blog
draft: false
---

模型需要加载进显存的参数量取决于两个变量：**推理/训练** 和 **模型权重**。

其中，模型权重的计算参考 [模型参数量的计算：Attention与卷积层](/posts/model-weights)。

## 推理时的显存占用

推理时，模型参数量 $\approx$ 模型权重。

> [!NOTE] KV Cache
> 对于 Transformer，我们会有**推理加速**机制，即 **KV Cache**，它会大量占用显存，无法忽略。
>
> **原理**：每轮推理时，都会缓存每个 token 的 $v_k$, $v_v$，用以下次计算 attention。
> - 能这样做，是因为推理时模型权重冻结，故对于每个 token 会算出相同的 k, v。
> - 不缓存 q，是因为后续每次推理的 token 肯定都要和上文的 k, v 交互，但不会与上文的 q 交互。
>
> 计算如下：
>
> $$
> \text{KVCache} = 2 \times b \times s_{\text{total}} \times L \times H_{\text{kv}} \times d_{\text{head}} \times \text{bytes\_per\_elem}
> $$
>
> - $b$: batch size
> - $s_{\text{total}}$: prompt + 已生成 token 总数
> - $L$: 层数
> - $H_{\text{kv}}$: KV 头数（GQA 中可能小于查询头数）
> - $d_{\text{head}}$: 每头维度
> - `bytes_per_elem` 通常为 2（FP16）

## 训练时的显存占用

使用 **混合精度 + Adam 优化器** 时，常驻显存为：

| 项目 | 大小 |
| ---- | ---- |
| 模型权重（前向） | P |
| 梯度 | P |
| 梯度副本 | P |
| 优化器一阶矩 $m$ | P |
| 优化器二阶矩 $v$ | P |
| **基础占用合计** | **5P** |

其中 P 为模型参数量。

### LoRA / QLoRA（参数高效微调 PEFT）

- 只在微调时使用，因为它有低秩假设，即认为参数只在参数空间的低秩子空间变动。
- 微调时，模型参数是冻结的，不过仍然需要载入。
- 它的主要优化点其实是优化了优化器维护参数，因为我们实际训练的是 LoRA，一个小参数模型，其对应的优化器维护参数自然也就很小。
- 假设 LoRA 权重为 $p$，则总参数量为 $P + 4p$，其中 $p \ll P$。
