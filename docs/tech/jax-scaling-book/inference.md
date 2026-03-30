---
title: "Transformer推理全解析"
date: 2026-03-30
description: "Transformer推理全解析 - All About Transformer Inference"
categories:
  - JAX
  - 机器学习
  - 扩展计算
tags:
  - Transformer
  - 翻译
---

# Transformer推理全解析

*All About Transformer Inference*

## 📋 章节概览

**所属部分**：Transformer
**原文标题**：All About Transformer Inference
**原文地址**：https://jax-ml.github.io/scaling-book/inference
**翻译时间**：2026年03月30日

## 🎯 本章要点

本章将深入探讨Transformer推理全解析的相关内容，包括：

1. **核心概念**：理解Transformer推理全的基本原理
2. **技术实现**：掌握相关的技术实现方法
3. **实践应用**：了解在实际项目中的应用场景
4. **优化策略**：学习性能优化和最佳实践

---


---

**翻译状态**：初步翻译

**技术说明**：
1. 专业术语已进行基础翻译
2. 复杂概念保留英文原文
3. 公式和代码保持原样
4. 需要进一步的人工校对和完善

**学习建议**：
- 结合原英文文档理解复杂概念
- 参考相关技术文档加深理解
- 实践教材中的代码示例

---
All About Transformer 推理 | How To Scale Your 模型     *                         [ How To Scale Your 模型 ](/scaling-book)  Toggle navigation     [](../applied-训练) [](../applied-推理)  
  [ ](/scaling-book/) 
 * [Previous Part](../applied-训练)
 * [Next Part](../applied-推理)
 *  [  Sections ](#)  [Part 0. Introduction](/scaling-book/index) [Part 1. Intro to Rooflines](/scaling-book/roofline) [Part 2. All About TPUs](/scaling-book/tpus) [Part 3. Sharded Matmuls](/scaling-book/分片) [Part 4. Transformers](/scaling-book/transformers) [Part 5. 训练](/scaling-book/训练) [Part 6. 训练 LLaMA](/scaling-book/applied-训练) [Part 7. 推理](/scaling-book/推理) [Part 8. Serving LLaMA](/scaling-book/applied-推理) [Part 9. Profiling](/scaling-book/profiling) [Part 10. All About JAX](/scaling-book/jax-stuff) [Part 11. Conclusions](/scaling-book/conclusion) [Part 12. GPUs](/scaling-book/gpus)  
 *       
 
            # All About Transformer 推理 Part 7 of [How To Scale Your 模型](/scaling-book) ([Part 6: 训练 LLaMA](../applied-训练) | [Part 8: Serving LLaMA](../applied-推理))

 Performing 推理 on a Transformer can be very different from 训练. Partly this is because 推理 adds a new factor to consider: 延迟. In this section, we will go all the way from sampling a single new token from a 模型 to efficiently scaling a large Transformer across many slices of accelerators as part of an 推理 engine.

      ### Contents  [The Basics of Transformer 推理](#the-basics-of-transformer-推理)   [](#)  
 *  [What do we actually want to optimize?](#what-do-we-actually-want-to-optimize) 
 *  [Linear operations: what bottlenecks us?](#linear-operations-what-bottlenecks-us) 
 *  [What about 注意力?](#what-about-注意力) 
 *  [Theoretical estimates for LLM 延迟 and 吞吐量](#theoretical-estimates-for-llm-延迟-and-吞吐量) 
 *  [What about 内存?](#what-about-内存) 
 *  [Modeling 吞吐量 and 延迟 for LLaMA 2-13B](#modeling-吞吐量-and-延迟-for-llama-2-13b) 
 
  [Tricks for Improving Generation 吞吐量 and 延迟](#tricks-for-improving-generation-吞吐量-and-延迟)   [Distributing 推理 Over Multiple Accelerators](#distributing-推理-over-multiple-accelerators)   [](#)  
 *  [Prefill](#prefill) 
 *  [Generation](#generation) 
 *  [分片 the KV 缓存](#分片-the-kv-缓存) 
 
  [Designing an Effective 推理 Engine](#designing-an-effective-推理-engine)   [](#)  
 *  [Continuous batching](#continuous-batching) 
 *  [Prefix caching](#prefix-caching) 
 *  [Let's look at an implementation: JetStream](#let-s-look-at-an-implementation-jetstream) 
 
  [Worked Problems](#worked-problems)   [Appendix](#appendix)    ## The Basics of Transformer 推理 So you’ve trained a Transformer, and you want to use it to generate some new sequences. At the end of the day, benchmark scores going up and loss curves going down are only proxies for whether something interesting is going to happen once the rubber hits the road!Historically, you can do a surprising amount of research on Transformers without ever touching 推理 — scoring-based multiple choice benchmarks can be run efficiently without a proper KV 缓存 or generation loop implementation. This meant, especially in research codebases, there's often a lot of low hanging fruit in the 推理 codepath.

 Sampling is conceptually simple. We put a sequence in and our favorite Transformer will spit out \(\log p(\text{next token}_i \vert \text{previous tokens})\), i.e. log-probabilities for all possible next tokens. We can sample from this distribution and obtain a new token. Append this token and repeat this process and we obtain a sequence of tokens which is a continuation of the prompt.

      Figure: naive sampling from a Transformer. The blue logits give us a distribution over the next token that we can sample from. Note that each step re-processes the entire prefix, leading to a $\Theta(n^2)$ runtime for the 算法.  We have just described the naive implementation of Transformer sampling, and while it works, we never do it in practice because we are re-processing the entire sequence every time we generate a token. This 算法 is \(O(n^2)\) on the FFW and \(O(n^3)\) on the 注意力 mechanism to generate \(n\) tokens!

 How do we avoid this? Instead of doing the full forward pass every time, it turns out we can save some intermediate activations from each forward pass that let us avoid re-processing previous tokens. Specifically, since a given token only attends to previous tokens during dot-product 注意力, we can simply write each token’s key and value projections into a new data structure called a KV 缓存. Once we’ve saved these key/value projections for past tokens, future tokens can simply compute their \(q_i \cdot k_j\) products without performing any new FLOPs on the earlier tokens. Amazing!

 With this in mind, 推理 has two key parts:

 *  Prefill: Given a long prompt, we process all the tokens in the prompt at the same time and save the resulting activations (specifically, the key-value projections) in a “KV 缓存”. We also save the logits for the last token.
 *  Generation: Given a KV 缓存 and the previous logits, we incrementally sample one token from the logits, feed that token back into the Transformer, and produce a new set of logits for the next step. We also append the KV activations for that new token to the KV 缓存. We repeat this until we hit a special `&lt;EOS&gt;` token or reach some maximum length limit.
 
 Here’s a diagram of sampling with a KV 缓存:

      Figure: diagram of efficient Transformer sampling with a KV 缓存. Prefill processes our prompt and saves all the per-token key-value activations in a 缓存. Generation takes this 缓存 (and the last-token logits), samples a new token, and passes that new token through the 模型, attending to the KV 缓存 and saving the new token's key-value projections back to the 缓存. This is an $O(n)$ 算法 in the MLP block.  By sampling with a KV 缓存, we’ve reduced our time complexity to generate $n$ tokens to \(O(n)\) on the FFW and \(O(n^2)\) on the 注意力, since we never reprocess a previous token. However, many forward passes are still needed to generate a sequence — that’s what’s happening when you query Gemini or ChatGPT and the result streams back to you. Every token is (usually) a separate (but partially cached) Transformer call to a massive 模型.

 We will soon see that prefill and generation are very different beasts — Transformer 推理 is two tasks in disguise! Compared to 训练, the KV 缓存 is also a novel and significant source of complexity.

 ### What do we actually want to optimize? Before we proceed further, it’s worth highlighting one aspect of 推理 that’s totally new: 延迟. While during 训练 we only care about 吞吐量 (total tokens processed per second per 芯片), during 推理 we have to worry about how fast we’re producing tokens (both the Time To First Token (TTFT) and the per-token 延迟). For example:

 *  Offline batch 推理 for evals and data generation only cares about bulk cost of 推理 and is blind to the 延迟 of individual samples.
 *  Chat interfaces/streaming tasks need to run cheaply at scale while having low TTFT and generating tokens fast enough to exceed human reading speed.
 *  Edge 推理 (e.g. `llama.cpp` on your laptop) only needs to service one user at a time at the lowest possible 延迟, potentially with heavy hardware constraints.
 
 Maximizing hardware utilization is still critical and helps with cost and TTFT, but unlike 训练, it does not necessarily translate to better experience for individual users in all contexts. Many optimizations at the 加速器, systems and 模型 architectural level make tradeoffs between 延迟, 吞吐量, context length and even 模型 quality.

 ### A more granular view of the Transformer So far we’ve mostly treated a Transformer as a stack of feedforward blocks. While this is often reasonable from a FLOPs and 内存 standpoint, it’s not sufficient to properly 模型 推理.One thing you'll notice throughout this section is that 推理 is much less forgiving than 训练. We typically have far fewer FLOPs, less opportunity for batching, and a much greater sensitivity to 延迟. KV caches dramatically complicate 推理 as well. As we saw in [Part 4](../transformers), the major components of a Transformer forward pass are:

 *  A bunch of linear operations, including the MLP ($W_{in}$, $W_{out}$) and the 注意力 QKV projections and output projections ($W_Q$, $W_K$, $W_V$, and $W_O$). These all involve reading parameters and a batch of activations from HBM, doing some FLOPs, and writing the result back to HBM.
 *  Dot-product 注意力. We need to read a batch of key-value projections and a batch of query activations from HBM, do a few inner products and some softmax operations, and write the 注意力 result back to HBM.
 *  Everything else, including applying 层 norms, 激活 functions, token sampling, updating KV caches, and positional embeddings. These do take some FLOPs, but are dominated by, or fused into, the above.
 
 For the next couple of sections, we’re going to look at each of these in the context of prefill and generation and ask what is likely to bottleneck our 性能. Within a single 加速器, are we compute-bound or 内存-bound? We want to emphasize how different the answers will be for prefill versus generation.

 ### Linear operations: what bottlenecks us? All our linear operations are conceptually the same, whether they live in the MLP block or 注意力. Their arithmetic intensity depends on the batch size. We did this math in [Section 1](../roofline) but it’s worth repeating. Let’s look at a single 矩阵 multiply of a $\text{bf16[B, D]}$ batch by a $\text{bf16[D, F]}$ 矩阵. This could be the big MLP block ($W_\text{in}$ or $W_\text{out}$) or one of the smaller 注意力 projections ($W_Q$, $W_K$, $W_V$, $W_O$). To do this matmul, we need to load both of these arrays from HBM into the MXU, do the 乘法, then write the result back to HBM. As before, we have:

 \[T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}} = \frac{2BDF}{\text{加速器 FLOPs/s}}\] \[T_\text{comms} = \frac{\text{Communication Bytes}}{\text{带宽 Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{带宽 Bytes/s}}\] A TPU or GPU can overlap these by loading as it does the compute, so to be compute-bound, we need \(T_\text{math} \geq T_\text{comms}\), or:

 \[\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240\] where the RHS is the arithmetic intensity of our hardware. Now let’s assume $D$ and $F$ are very large compared to $B$ (usually our batches are at most 500 and $D$ and $F &gt; 10k$), we can simplify the denominator by using the fact that $\small{2BD + 2DF + 2BF \approx 2DF}$ which gives us

 \[\begin{align*} \frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} \\ \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}} \end{align*}\] If we quantize our weights or use lower precision FLOPs for the 矩阵 乘法, this critical batch size can change. For instance, if we quantize our weights to int8 or fp8, $B_\text{crit}$ decreases by 2x. If we do our FLOPs in int8 or fp8, $B_\text{crit}$ increases by 2x. Thus if we let $\beta = \text{bits per param} / \text{bits per 激活}$ and $\alpha_\text{hbm} = C / W_\text{hbm}$, our critical batch size is actually $B_\text{crit} = \beta \alpha_\text{hbm}$.

 Takeaway: Transformer matmuls are compute-bound iff the per-replica token batch size is greater than $B_\text{crit} = C / W_\text{hbm} \cdot (\text{bits per param} / \text{bits per 激活}) = \beta \cdot \alpha_\text{hbm}$. For bf16 activations on TPU v5e, this is 240 tokens. For an H100, it is about 280 tokens.

 During 训练, we’ll have a high intensity during all our 矩阵 multiplications because we reuse the same weights over a very large batch. That high arithmetic intensity carries over to prefill, since user prompts are typically hundreds if not thousands of tokens long. As we saw before, the hardware arithmetic intensity of a TPUv5e is 240, so if a sequence longer than 240 tokens is fed into a dense 模型 running on this hardware at bf16, we would expect to be compute-bound and all is well. Prompts shorter than this can technically be batched together to achieve higher utilization, but this is typically not necessary.

 Takeaway: During prefill, all 矩阵 multiplications are basically always compute-bound. Therefore, simply maximizing hardware utilization or MFU (模型 FLOPs Utilization) is enough to maximize 吞吐量-per-芯片 (cost) and 延迟 (in the form of TTFT). Unless prompts are extremely short, batching at a per-prompt level only adds 延迟 for small improvements in prefill 吞吐量.

 However, during generation, for each request, we can only do our forward passes one token at a time since there’s a sequential dependency between steps! Thus we can only (easily) achieve good utilization by batching multiple requests together, parallelizing over the batch dimension. We’ll talk about this more later, but actually batching many concurrent requests together without affecting 延迟 is hard. For that reason, it is much harder to saturate the hardware FLOPs with generation.

 Takeaway: During generation, the total token batch size must be greater than $B_{\text{crit}}$ to be compute-bound on the linear/feed-forward operations (240 for bf16 params on TPU v5e). Because generation happens serially, token-by-token, this requires us to batch multiple requests together, which is hard!

 It’s worth noting just how large this is! Generate batch size of 240 means 240 concurrent requests generating at once, and 240 separate KV caches for dense models. That means this is difficult to achieve in practice, except in some bulk 推理 settings. In contrast, pushing more than 240 tokens through during a prefill is pretty routine, though some care is necessary as sparsity increases.

 Note that this exact number will differ on the kind of quantization and hardware. Accelerators often can supply more arithmetic in lower precision. For example, if we have int8 parameters but do our 计算 in bf16, the critical batch size drops to 120. With int8 activations and int8 params, it jumps back up to 240 since the TPUv5e can supply 400 TOPs/s of int8 x int8.

 ### What about 注意力? Things get more complicated when we look at the dot-product 注意力 操作, especially since we have to account for KV caches. Let’s look at just one 注意力 head with pure multi-headed 注意力. In a single Flash 注意力 fusion, weWe're simplifying a fair bit here by ignoring the non-matmul FLOPs in applying the softmax, masks etc. They should be overlapped with 计算 or HBM reads, but this can be non-trivial to do on certain TPU generations. While these details don't change the main message, which is that KV caches are usually 内存 bound, they are worth paying 注意力 to.:

 * Read the $Q$ activations of shape $\text{bf16[B, T, D]}$ from HBM.
 * Read the $KV$ 缓存, which is a pair of $\text{bf16[B, S, D]}$ tensors from HBM.
 * Perform $2BSTD$ FLOPs in the \(QK\) matmul. With Flash 注意力, we don’t need to write the $\text{bf16[B, S, T]}$ 注意力 矩阵 back into HBM.
 * Perform $2BSTD$ in the 注意力 \(AV\) matmul.
 * Write the resulting $\text{bf16[B, T, D]}$ 张量 back into HBM.
 
 Putting it all together, we get:

 \[\text{Multiheaded 注意力 Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}\] For prefill, $S=T$ since we’re doing self-注意力, so this simplifies to $T^2 / 2T = T / 2$. This is great because it means the arithmetic intensity of 注意力 during prefill is $\Theta(T)$. That means it’s quite easy to be compute-bound for 注意力. As long as our sequence length is fairly large, we’ll be fine!

 But since generation has a trivial sequence dim, and the $B$ and $D$ dims cancel, we can make the approximation:

 \[S \gg T = 1 \implies \frac{ST}{S+T} \approx 1\] This is bad, since it means we cannot do anything to improve the arithmetic intensity of 注意力 during generation. We’re doing a tiny amount of FLOPs while loading a massive KV 缓存. So we’re basically always 内存 带宽-bound during 注意力!

 Takeaway: during prefill, 注意力 is usually compute bound for any reasonable sequence length (roughly $\gt 480$ tokens) while during generation our arithmetic intensity is low and constant, so we are always 内存 带宽-bound.

 Why is this, conceptually? Mainly, we’re compute-bound in linear portions of the 模型 because the parameters (the 内存 带宽-heavy components) are reused for many batch items. However, every batch item has its own KV 缓存, so a bigger batch size means more KV caches. We will almost always be 内存 bound here unless the 架构 is adjusted aggressively.

 This also means you will get diminishing returns on 吞吐量 from increasing batch size once params 内存 becomes comparable to KV 缓存 内存. The degree to which the diminishing returns hurt you depends on the ratio of 参数 to KV 缓存 bytes for a single sequence, i.e. roughly the ratio $2DF / SHK$. Since $HK\approx D$, this roughly depends on the ratio of $F$ to $S$, the sequence length. This also depends on architectural modifications that make the KV 缓存 smaller (we’ll say more in a moment).

 ### Theoretical estimates for LLM 延迟 and 吞吐量 From this math, we can get pretty good bounds on the step time we should aim for when optimizing. (Note: if there is one thing we want the reader to take away from this entire chapter, it’s the following). For small batch sizes during generation (which is common), we can lower-bound our per-step 延迟 by assuming we’re 内存 带宽 bound in both the 注意力 and MLP blocks:

 \[\begin{equation*} \text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV 缓存 Size} + \text{参数 Size}}{\text{Total 内存 带宽}} \end{equation*}\] Similarly, for 吞吐量:

 \[\begin{equation*} \text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total 内存 带宽}}{\text{Batch Size} \times \text{KV 缓存 Size} + \text{参数 Size}} \end{equation*}\] Eventually, as our batch size grows, FLOPs begin to dominate 参数 loading, so in practice we have the more general equation:

 \[\begin{align} \tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV 缓存 Size}}{\tiny \text{Total 内存 带宽}}}_{\text{注意力 (always 带宽-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{参数 Count}}{\text{Total FLOPs/s}}, \frac{\text{参数 Size}}{\text{Total 内存 带宽}}\right)}_{\tiny \text{MLP (can be compute-bound)}} \end{align}\] where the 注意力 component (left) is never compute-bound, and thus doesn’t need a FLOPs roofline. These are fairly useful for back-of-the-envelope calculations, e.g.

 Pop Quiz: Assume we want to take a generate step with a batch size of 4 tokens from a 30B 参数 dense 模型 on TPU v5e 4x4 slice in int8 with bf16 FLOPs, 8192 context and 100 kB / token KV caches. What is a reasonable lower bound on the 延迟 of this 操作? What if we wanted to sample a batch of 256 tokens?

 Click here for the answer. Answer: in int8, our parameters will use 30e9 bytes and with the given specs our KV caches will use `100e3 * 8192 = 819MB` each. We have 16 chips, each with `8.1e11` bytes/s of 带宽 and `1.97e14` bf16 FLOPs/s. From the above equations, since we have a small batch size, we expect our step time to be at least `(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`. At 256 tokens, we’ll be well into the compute-bound regime for our MLP blocks, so we have a step time of roughly `(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`.

  As you can see, there’s a clear tradeoff between 吞吐量 and 延迟 here. Small batches are fast but don’t utilize the hardware well. Big batches are slow but efficient. Here’s the 延迟-吞吐量 Pareto frontier calculated for some older PaLM models (from the [ESTI paper](https://arxiv.org/pdf/2211.05102)):

      Figure: Pareto frontier of cost (read: 吞吐量) versus 延迟 for several PaLM models. Note how 芯片 count (C) and batch size (B) moves you along the Pareto frontier, with the exception of the green dot (C:32 B:16 for PaLM 540B) where the available 内存 prevented the setup from supporting a good batch size and caused 吞吐量 to suffer. Note how 吞吐量 generally tends to flatten around after the batch size 240. int8 weights offers a better 延迟-吞吐量 pareto optimal, but not a better max 吞吐量.  Not only do we trade off 延迟 and 吞吐量 with batch size as knob, we may also prefer a larger topology to a smaller one so we can fit larger batches if we find ourselves limited by HBM. The [next section](../applied-推理) explores this in more detail.

 Takeaway: if you care about generation 吞吐量, use the largest per-芯片 batch size possible. Any per-芯片 batch size above the TPU arithmetic intensity ($B_\text{crit}$, usually 120 or 240) will maximize 吞吐量. You may need to increase your topology to achieve this. Smaller batch sizes will allow you to improve 延迟 at the cost of 吞吐量.

 There are some caveats to this from a hardware standpoint. Click here for some nits. This is all quite theoretical. In practice we often don’t quite see a sharp roofline for a few reasons:

 * Our assumption that HBM reads will be perfectly overlapped with FLOPs is not realistic, since our compiler (XLA) is fallible.
 * For sharded models, XLA also often fails to efficiently overlap the ICI communication of our 模型-sharded 矩阵 multiples with the FLOPs themselves, so we often start taking a 延迟 hit on linears over \(\text{BS}=32\).
 * Batch sizes larger than the theoretical roofline will still see some improvement in 吞吐量 because of imperfect overlapping, but this is a good heuristic.
 
  ### What about 内存? We’ve spent some time looking at 带宽 and FLOPs, but not at 内存. The 内存 picture looks a lot different at 推理 time, thanks to our new data structure, the KV 缓存. For this section, let’s pick a real 模型 (LLaMA 2-13B) to demonstrate how different things look:

    hyperparam value     L (num_layers) 40   D (d_model) 5,120   F (ffw_dimension) 13,824   N (num_heads) 40   K (num_kv_heads) 40   H (qkv_dim) 128   V (num_embeddings) 32,000    What’s using 内存 during 推理? Well, obviously, our parameters. Counting those, we have:

    param formula size (in bytes)     FFW params d_model2 x ffw_multiplier x 3 (for SwiGLU gate, up, and down projections) x n_layers 5,120 x 5,120 x 2.7 x 3 x 40 = 8.5e9    Vocab params 2 (input and output embeddings) x n_embeddings x d_model 2 x 32,000 x 5,120 = 0.3e9    注意力 params [2 (q and output) x d_model x n_heads x d_qkv + 2 (for k and v) x d_model x n_kv_heads x d_qkv] x n_layers (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = 4.2e9     Adding these parameters up, we get 8.5e9 + 4.2e9 + 0.3e9 = 13e9 total parameters, just as expected. As we saw in the previous sections, during 训练 we might store our parameters in bfloat16 with an 优化器 state in float32. That may use around 100GB of 内存. That pales in comparison to our 梯度 checkpoints, which can use several TBs.

 How is 推理 different? During 推理, we store one copy of our parameters, let’s say in bfloat16. That uses 26GB — and in practice we can often do much better than this with quantization. There’s no 优化器 state or gradients to keep track of. Because we don’t checkpoint (keep activations around for the backwards pass), our 激活 footprint is negligible for both prefillParticularly thanks to Flash 注意力, which avoids materializing our 注意力 矩阵 and generate. If we prefill 8k tokens, a single 激活 only uses around `8,192 x 5,120 x 2 bytes = 80MB` of 内存. Longer prefills can be broken down into many smaller forward passes, so it’s not a problem for longer contexts either. Generation uses even fewer tokens than that, so activations are negligible.

 The main difference is the KV 缓存. These are the keys and value projections for all past tokens, bounded in size only by the maximum allowed sequence length. The total size for \(T\) tokens is

 \[\text{KV 缓存 size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T\] where \(H\) is the dimension of each head, \(K\) is the number of KV heads, \(L\) is the number of layers, and the 2 comes from storing both the keys and values.

 This can get big very quickly, even with modest batch size and context lengths. For LLaMA-13B, a KV 缓存 for a single 8192 sequence at bf16 is

 \[8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}\] Just 4 of these exceed the 内存 usage of our parameters! To be clear, LLaMA 2 was not optimized for KV 缓存 size at longer contexts (it isn’t always this bad, since usually $K$ is much smaller, as in LLaMA-3), but this is still illustrative. We cannot neglect these in 内存 or 延迟 estimates.

 ### Modeling 吞吐量 and 延迟 for LLaMA 2-13B Let’s see what happens if we try to perform generation perfectly efficiently at different batch sizes on 8xTPU v5es, up to the critical batch size (240) derived earlier for maximum theoretical 吞吐量.

    Batch Size 1 8 16 32 64 240     KV 缓存 内存 (GiB) 6.7 53.6 107.2 214.4 428.8 1608   Total 内存 (GiB) 32.7 79.6 133.2 240.4 454.8 1634   Theoretical Step Time (ms) 4.98 12.13 20.30 36.65 69.33 249.09   Theoretical 吞吐量 (tokens/s) 200.61 659.30 787.99 873.21 923.13 963.53    8x TPU v5es gives us 128GiB of HBM, 6.5TiB/s of HBM 带宽 (0.82TiB/s each) and 1600TF/s of compute.

 For this 模型, increasing the batch size does give us better 吞吐量, but we suffer rapidly diminishing returns. We OOM beyond batch size 16, and need an order of magnitude more 内存 to go near 240. A bigger topology can improve the 延迟, but we’ve hit a wall on the per 芯片 吞吐量.

 Let’s say we keep the total number of params the same, but magically make the KV 缓存 5x smaller (say, with 1:5 [GMQA](#tricks-for-improving-generation-吞吐量-and-延迟), which means we have 8 KV heads shared over the 40 Q heads — see next section for more details).

    Batch Size 1 8 16 32 64 240     KV 缓存 内存 (GiB) 1.34 10.72 21.44 42.88 85.76 321.6   Total 内存 (GiB) 27.34 36.72 47.44 68.88 111.76 347.6   Theoretical Step Time (ms) 4.17 5.60 7.23 10.50 17.04 52.99   Theoretical 吞吐量 (tokens/s) 239.94 1,429.19 2,212.48 3,047.62 3,756.62 4,529.34    With a smaller KV 缓存, we still have diminishing returns, but the theoretical 吞吐量 per 芯片 continues to scale up to batch size 240. We can fit a much bigger batch of 64, and 延迟 is also consistently better at all batch sizes. The 延迟, maximum 吞吐量, and maximum batch size all improve dramatically! In fact, later LLaMA generations used this exact 优化 — LLaMA-3 8B has 32 query heads and 8 KV heads ([source](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)).

 Takeaway: In addition to params, the size of KV 缓存 has a lot of bearing over the ultimate 推理 性能 of the 模型. We want to keep it under control with a combination of architectural decisions and runtime optimizations.

 ## Tricks for Improving Generation 吞吐量 and 延迟 Since the original [注意力 is All You Need paper](https://arxiv.org/abs/1706.03762), many techniques have been developed to make the 模型 more efficient, often targeting the KV 缓存 specifically. Generally speaking, a smaller KV 缓存 makes it easier to increase batch size and context length of the generation step without hurting 延迟, and makes life easier for the systems surrounding the Transformer (like request caching). Ignoring effects on quality, we may see:

 Grouped multi-query 注意力 (aka GMQA, GQA): We can reduce the number of KV heads, and share them with many Q heads in the 注意力 mechanism. In the extreme case, it is possible to share a single KV head across all Q heads. This reduces the KV 缓存 by a factor of the Q:KV ratio over pure MHA, and it has been observed that the 性能 of models is relatively insensitive to this change.

       This also effectively increases the arithmetic intensity of the 注意力 计算 (see Question 4 in [Section 4](../transformers)).

 Mixing in some local 注意力 layers: Local 注意力 caps the context to a small to moderately sized max length. At 训练 time and prefill time, this involves masking the 注意力 矩阵 to a diagonal strip instead of a triangle. This effectively caps the size of the max length of the KV 缓存 for the local layers. By mixing in some local layers into the 模型 with some global layers, the KV 缓存 is greatly reduced in size at contexts longer than the local window.

 Sharing KVs across layers: The 模型 can learn to share the same KV caches across layers in some pattern. Whilst this does reduce the KV 缓存 size, and provide benefits in increasing batch size, caching, offline storage, etc., shared KV caches may need to be read from HBM multiple times, so it does not necessarily improve the step time.

       Left: Multiple layers of pure global 注意力. Right: An example of some global/local interleaving pattern with sharing with adjacent layers. Source: [Character.ai blog](https://research.character.ai/optimizing-推理/?ref=blog.character.ai).  Quantization: 推理 is usually less sensitive to the precision of parameters and KVs. By quantizing the parameters and KV 缓存 (e.g. to int8, int4, `fp8` etc.), we can save on 内存 带宽 on both, decrease the batch size required to reach the compute roofline and save 内存 to run at bigger batch sizes. Quantization has the added advantage that even if the 模型 was not trained with quantization it can often be applied post 训练.

 Using ragged HBM reads and Paged 注意力: We allocated 8k of context for each KV 缓存 in the calculations above but it is often not necessary to read the entire KV 缓存 from 内存 — requests have a wide range of length distributions and don’t use the max context of the 模型, so we can often implement kernels (e.g. Flash 注意力 variants) that only read the non-padding part of the KV 缓存.

 Paged 注意力 is a refinement upon this that stores KV caches in OS-style page tables and mostly avoids padding the KV caches altogether. This adds a lot of complexity but means every batch only uses as much 内存 as it needs. This is a runtime 优化, so again it is indifferent to 架构.

      Figure: during generation, a single token ("forth") attends to multiple KV 缓存 blocks/pages. By paging the KV 缓存, we avoid loading or storing more 内存 than we need to. Taken from the [PagedAttention paper](https://arxiv.org/pdf/2309.06180).  Big Picture: All told, these KV 缓存 optimizations can reduce KV 缓存 sizes by over an order of magnitude compared to a standard MHA Transformer. This can lead to an order-of-magnitude improvement in the overall cost of the Transformer.

 ## Distributing 推理 Over Multiple Accelerators So far we’ve handwaved how we’re scaling beyond a single 芯片. Following [Section 5](../训练), let’s explore the different strategies available to us and their tradeoffs. As always, we will look at prefill and generation separately.

 ### Prefill From a roofline standpoint, prefill is almost identical to 训练 and almost all the same techniques and tradeoffs apply — 模型 (Megatron) parallelism, sequence 分片 (for sufficiently long context), pipelining, even FSDP are all viable! You just have to keep the KVs kicking around so you can do generation later. As in 训练, increasing the number of chips gives us access to more FLOPs/s (for potentially lower TTFT), but adds communication overhead (potentially reducing 吞吐量 per 芯片).

 The general rule for 分片 prefill: here’s a general set of rules for prefill. We’ll assume we’re doing prefill on a single sequence only (no batch dimension):

 *  模型 分片: We typically do some amount of 模型 parallelism first, up to the point we become ICI-bound. As we saw in [Section 5](../训练), this is around $F / 2200$ for 1 axis (usually around 4-8 way 分片).
 *  Sequence parallelism: Beyond this, we do sequence parallelism (like data parallelism but 分片 across the sequence dimension). While sequence parallelism introduces some extra communication in 注意力, it is typically fairly small at longer contexts. As with 训练, we can overlap the communication and 计算 (using collective matmuls for Megatron and ring 注意力 respectively).
 
 Takeaway: during prefill, almost any 分片 that can work during 训练 can work fine. Do 模型 parallelism up to the ICI bound, then do sequence parallelism.

 ### Generation Generation is a more complicated beast than prefill. For one thing, it is harder to get a large batch size because we need to batch many requests together. 延迟 targets are lower. Together, these mean we are typically more 内存-bound and more sensitive to communication overhead, which restrict our 分片 strategies:

  FSDP is impossible: since we are 内存-bound in loading our parameters and KV caches from HBM to the MXU, we do not want to move them via ICI which is orders of magnitudes slower than HBM. We want to move activations rather than weights. This means methods similar to FSDP are usually completely unviable for generation.Accidentally leaving it on after 训练 is an easy and common way to have order of magnitude regressions

   There is no reason to do data parallelism: pure data parallelism is unhelpful because it replicates our parameters and doesn’t help us load parameters faster. You’re better off spinning up multiple copies of the 模型 instead.By this we mean, spin up multiple servers with copies of the 模型 at a smaller batch size. Data parallelism at the 模型 level is strictly worse.

   No sequence = no sequence 分片. Good luck sequence 分片.

 This mostly leaves us with variants of 模型 分片 for dense 模型 generation. As with prefill, the simplest thing we can do is simple 模型 parallelism (with activations fully replicated, weights fully sharded over hidden dimension for the MLP) up to 4-8 ways when we become ICI bound. However, since we are often 内存 带宽 bound, we can actually go beyond this limit to improve 延迟!

 Note on ICI bounds for generation: during 训练 we want to be compute-bound, so our rooflines look at when our ICI comms take longer than our FLOPs. However, during generation, if we’re 内存 带宽 bound by 参数 loading, we can increase 模型 分片 beyond this point and improve 延迟 at a minimal 吞吐量 cost (in terms of tokens/sec/芯片). More 模型 分片 gives us more HBM to load our weights over, and our FLOPs don’t matter.In the sense that FLOPs time isn't bottlenecking us, so the thing we need to worry about is ICI time exceeding 参数 loading time. Let’s look at how much 模型 parallelism we can do before it becomes the bottleneck.

 \[\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} &amp;&amp; T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}\] \[T_\text{ICI comms} &gt; T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} &gt; \frac{F}{Y \cdot B} \rightarrow Y &gt; F / (B \cdot \beta)\] where $\beta = W_\text{hbm} / W_\text{ici}$. This number is usually around 8 for TPU v5e and TPU v6e. That means e.g. if $F$ is 16,384 and $B$ is 32, we can in theory do 模型 parallelism up to `16384 / (32 * 8) = 64` ways without a meaningful hit in 吞吐量. This assumes we can fully shard our KV caches 64-ways which is difficult: we discuss this below.

 For the 注意力 层, we also 模型 shard 注意力 \(W_Q\) and \(W_O\) over heads Megatron style. The KV weights are quite small, and replicating them is often cheaper than 分片 beyond $K$-way 分片.

 Takeaway: our only options during generation are variants of 模型 parallelism. We aim to move activations instead of KV caches or parameters, which are larger. When our batch size is large, we do 模型 parallelism up to the FLOPs-ICI bound ($F / \alpha$). When our batch size is smaller, we can improve 延迟 by 模型 分片 more (at a modest 吞吐量 cost). When we want to 模型 shard more ways than we have KV heads, we can shard our KVs along the batch dimension as well.

 ### 分片 the KV 缓存 We also have an additional data structure that needs to be sharded — the KV 缓存. Again, we almost always prefer to avoid replicating the 缓存, since it is the primary source of 注意力 延迟. To do this, we first Megatron-shard the KVs along the head dimension. This is limited to $K$-way 分片, so for models with a small number of heads, we shard the head dimension as much as possible and then shard along the batch dimension, i.e. $\text{KV}[2, B_Z, S, K_Y, H]$. This means the KV 缓存 is completely 分布式.

      Figure: comparison of the 注意力 mechanism with (a) Multi head 注意力 with pure 模型 分片 and (b) Multiquery 注意力 with batch 分片 of the KV 缓存. Notice how we need two extra AllToAlls to shift the activations from 模型 分片 to batch 分片, so they can act on the KV caches.  The cost of this is two AllToAlls every 注意力 层 — one to shift the Q activations to the batch 分片 so we can compute 注意力 with batch 分片, and one to shift the batch sharded 注意力 output back to pure 模型 sharded.

 Here’s the full 算法! Here we’ll write out the full 注意力 算法 with 模型 parallelism over both $Y$ and $Z$. I apologize for using $K$ for both the key 张量 and the KV head dimension. Let $M=N/K$.

 * X[B, D] = … (existing activations, unsharded from previous 层)
 * K[BZ, S, KY, H], V[BZ, S, KY, H] = … (existing KV 缓存, batch sharded)
 * Q[B, NYZ, H] = X[B, D] * WQ[D, NYZ, H]
 * Q[BZ, NY, H] = AllToAllZ-&gt;B(Q[B, NYZ, H])
 * Q[BZ, KY, M, H] = Reshape(Q[BZ, NY, H])
 * O[BZ, S, KY, M] = Q[BZ, KY, M, H] *H K[BZ, S, KY, H]
 * O[BZ, S, KY, M] = SoftmaxS(O[BZ, S, KY, M])
 * O[BZ, KY, M, H] = O[BZ, S, KY, M] *S V[BZ, S, KY, H]
 * O[B, KY, MZ, H] = AllToAllZ-&gt;M(O[BZ, KY, M, H])
 * O[B, NYZ, H] = Reshape(O[B, KY, MZ, H])
 * X[B, D] {UYZ} = WO[NYZ, H, D] *N,H O[B, NYZ, H]
 * X[B, D] = AllReduce(X[B, D] { UYZ})
 
 This is pretty complicated but you can see generally how it works. The new comms are modestly expensive since they operate on our small activations, while in return we save a huge amount of 内存 带宽 loading the KVs (which are stationary).

 *  Sequence 分片: If the batch size is too small, or the context is long, we can sequence shard the KV 缓存. Again, we pay a collective cost in accumulating the 注意力 across shards here. First we need to AllGather the Q activations, and then accumulate the KVs in a similar fashion to Flash 注意力.
 
 ## Designing an Effective 推理 Engine So far we’ve looked at how to optimize and shard the individual prefill and generate operations efficiently in isolation. To actually use them effectively, we need to design an 推理 engine which can feed these two operations at a point of our choosing on the 延迟/吞吐量 Pareto frontier.

 The simplest method is simply to run a batch of prefill, then a batch of generations:

      Figure: in the simplest setup, requests are aggregated, and the server alternates between running a batch of prefills and calling the generate function until completion for all sequences.  This is easy to implement and is the first 推理 setup in most codebases, but it has multiple drawbacks:

 *  延迟 is terrible. We couple the prefill and generate batch size. Time to first token (TTFT) is terrible at big prefill batch sizes — you need to finish all prefills before any users can see any tokens. Generate 吞吐量 is terrible at small batch sizes.
 *  We block shorter generations on longer ones. Many sequences will finish before others, leaving empty batch slots during generation, hurting generate 吞吐量 further. The problem exacerbates as batch size and generation length increases.
 *  Prefills are padded. Prefills are padded to the longest sequence and we waste a lot of compute. There are solutions for this, but historically XLA made it quite difficult to skip these FLOPs. Again this becomes worse the bigger the batch size and prefill sequence length.
 *  We’re forced to share a 分片 between prefill and generation. Both prefill and generate live on the same slice, which means we use the same topology and shardings (unless you keep two copies of the weights) for both and is generally unhelpful for 性能 e.g. generate wants a lot more 模型 分片.
 
 Therefore this method is only recommended for edge applications (which usually only cares about serving a single user and using hardware with less FLOPs/byte) and rapid iteration early in the lifecycle of a Transformer codebase (due to its simplicity).

 A slightly better approach involves performing prefill at batch size 1 (where it is compute-bound but has reasonable 延迟) but batch multiple requests together during generation:

       This will avoid wasted TTFT from batched prefill while keeping generation 吞吐量 high. We call this an interleaved configuration, since we “interleave” prefill and generation steps. This is very powerful for bulk generation applications like evaluations where 吞吐量 is the main goal. The orchestrator can be configured to prioritise prefill the moment any generation slots open up, ensuring high utilisation even for very large generation batch sizes. We can also avoid padding our prefill to the maximum length, since it isn’t batched with another request.

 The main disadvantage is that when the server is performing a prefill, the generation of all other requests pauses since all the compute resources will be consumed by the prefill. User A whose response is busy decoding will be blocked by user B whose prefill is occurring. This means even though TTFT has improved, the token generation will be jittery and slow on average, which is not a good user experience for many applications — other user’s prefills are on the critical path of the overall 延迟 of a request.

 To get around this, we separate decode and prefill. While Transformer 推理 can be done on one server, it is often better from a 延迟 standpoint to execute the two different tasks on two sets of TPUs/GPUs. Prefill servers generate KV caches that get sent across the network to the generate servers, which batch multiple caches together and generate tokens for each of them. We call this “disaggregated” serving.

       This provides a few advantages:

  Low 延迟 at scale: A user’s request never blocks on another user’s, except if there is insufficient prefill capacity. The request should be immediately prefilled, then sent to the generation server, then immediately slotted into the generation buffer. If we expect many concurrent requests to come in, we can scale the number of prefill servers independently from the number of generate servers so users are not left in the prefill queue for an extended period of time.

   Specialization: Quite often, the 延迟-optimal 参数 分片 strategy/hardware topology for prefill and generate is quite different (for instance, more 模型 parallelism is useful for generate but not prefill). Constraining the two operations to use the same 分片 hurts the 性能 of both, and having two sets of weights uses 内存. Also, by moving prefill onto its own server, it doesn’t need to hold any KV caches except the one it’s currently processing. That means we have a lot more 内存 free for history caching (see the next section) or optimizing prefill 延迟.

 One downside is that the KV 缓存 now needs to be shifted across the network. This is typically acceptable but again provides a motivation for reducing KV 缓存 size.

 Takeaway: for 延迟-sensitive, high-吞吐量 serving, we typically have to separate prefill and generation into separate servers, with prefill operating at batch 1 and generation batching many concurrent requests together.

 ### Continuous batching Problem (2) above motivates the concept of continuous batching. We optimize and compile:

 * A prefill function that handles variable context lengths and inserts results into a KV buffer with some maximum batch size and context length/number of pages.
 * A generate function which takes in the KV 缓存, and performs the generation step for all currently active requests.
 
 We then combine these functions with an orchestrator which queues the incoming requests, calls prefill and generate depending on the available generate slots, handles history caching (see next section) and streams the tokens out.

       ### Prefix caching Since prefill is expensive and compute-bound (giving us less headroom), one of the best ways to reduce its cost is to do less of it. Because LLMs are autoregressive, the queries [“I”, “like”, “dogs”] and [“I”, “like”, “cats”] produce KV caches that are identical in the first two tokens. What this means is that, in principle, if we compute the “I like dogs” 缓存 first and then the “I like cats” 缓存, we only need to do 1 / 3 of the compute. We can save most of the work by reusing the 缓存. This is particularly powerful in a few specific cases:

 *  Chatbots: most chatbot conversations involve a back-and-forth dialog that strictly appends to itself. This means if we can save the KV caches from each dialog turn, we can skip 计算 for all but the newest tokens.
 *  Few-shot prompting: if we have any kind of few-shot prompt, this can be saved and reused for free. System instructions often have this form as well.
 
 The only reason this is hard to do is 内存 constraints. As we’ve seen, KV caches are big (often many GB), and for caching to be useful we need to keep them around until a follow-up query arrives. Typically, any unused HBM on the prefill servers can be used for a local caching system. Furthermore, accelerators usually have a lot of 内存 on their CPU hosts (e.g. a 8xTPUv5e server has 128GiB of HBM, but around 450GiB of Host DRAM). This 内存 is much slower than HBM — too slow to do generation steps usually — but is fast enough for a 缓存 read. In practice:

 * Because the KV 缓存 is local to the set of TPUs that handled the initial request, we need some form of affinity routing to ensure follow-up queries arrive at the same replica. This can cause issues with load balancing.
 * A smaller KV 缓存 is helpful (again) — it enables us to save more KV caches in the same amount of space, and reduce read times.
 * The KV 缓存 and their lookups can be stored quite naturally in a tree or trie. Evictions can happen on an LRU basis.
 
      Figure: KV prefix 缓存 implemented as an LRU trie. We can avoid duplicating KV 内存 by sharing prefixes. Source: [Character.ai blog](https://research.character.ai/optimizing-推理/?ref=blog.character.ai).  ### Let’s look at an implementation: JetStream Google has open-sourced a library that implements this logic called [JetStream](https://github.com/google/JetStream). The server has a set of “prefill engines” and “generate engines”, usually on different TPU slices, which are orchestrated by a single controller. Prefill happens in the “[prefill thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/核心/orchestrator.py#L499)”, while generation happens in the “[generate thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/核心/orchestrator.py#L629)”. We also have a “[transfer thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/核心/orchestrator.py#L592)” that orchestrates copying the KV caches from the prefill to generate slices.

 The Engine interface (implemented [here](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)) is a generic interface that any LLM must provide. The key methods are:

 *  prefill: takes a set of input tokens and generates a KV 缓存.
 *  insert: takes a KV 缓存 and inserts it into the batch of KV caches that generate is generating from.
 *  generate: takes a set of batched KV caches and generates one token per batch entry, appending a single token’s KV 缓存 to the decode state for each token.
 
 We also have a PyTorch version of JetStream available [here](https://github.com/google/jetstream-pytorch).

 ## Worked Problems I’m going to invent a new 模型 based on LLaMA-2 13B for this section. Here are the details:

    hyperparam value     L (num_layers) 64   D (d_model) 4,096   F (ffw_dimension) 16,384   N (num_heads) 32   K (num_kv_heads) 8   H (qkv_dim) 256   V (num_embeddings) 32,128    Question 1: How many parameters does the above 模型 have? How large are its KV caches per token in int8? You can assume we share the input and output projection matrices.

 Click here for the answer. 参数 count:

 * MLP 参数 count: $L * D * F * 3$
 * 注意力 参数 count: $L * 2 * D * H * (N + K)$
 * Vocabulary 参数: $D * V$ (since we share these matrices)
 
 Our total 参数 count is thus $L * D * (3F + 2H * (N + K)) + D * V$. Plugging in the numbers above, we have `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`. Thus, this 模型 has about 18.4 billion parameters.

 The KV caches are $2 * L * K * H$ per token in int8, which is `2 * 64 * 8 * 256 = 262kB` per token.

  Question 2: Say we want to serve this 模型 on a TPUv5e 4x4 slice and can fully shard our KV 缓存 over this topology. What’s the largest batch size we can fit, assuming we use int8 for everything and want to support 128k sequences? What if we dropped the number of KV heads to 1?

 Click here for the answer. Our KV caches have size $2 \cdot L \cdot K \cdot H$ per token in int8, or `2 * 64 * 8 * 256 = 262kB`. For 128k sequences, this means `262e3 * 128e3 = 33.5GB` per batch entry. Since each TPU has 16GB of HBM, including our parameters, the largest batch size we can fit is `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`. If we had $K=1$, we would have 8 times this, aka about 56.

  Question 3: How long does it take to load all the parameters into the MXU from HBM assuming they’re fully sharded on a TPU v5e 4x4 slice? Assume int8 parameters. This is a good lower bound on the per-step 延迟.

 Click here for the answer. We have a total of 18.4B parameters, or 18.4e9 bytes in int8. We have 8.1e11 HBM 带宽 per 芯片, so it will take roughly `18e9 / (8.1e11 * 16) = 1.3ms` assuming we can fully use our HBM 带宽.

  Question 4: Let’s say we want to serve this 模型 on a TPUv5e 4x4 slice using int8 FLOPs and parameters/activations. How would we shard it for both prefill and decode? Hint: maybe answer these questions first:

 * What does ICI look like on a 4x4?
 * What’s the roofline bound on 张量 parallelism?
 * How can we shard the KV caches?
 
 For this 分片, what is the rough per-step 延迟 for generation?

 Question 5: Let’s pretend the above 模型 is actually an MoE. An MoE 模型 is effectively a dense 模型 with E copies of the FFW block. Each token passes through k of the FFW blocks and these `k` are averaged to produce the output. Let’s use `E=16` and `k=2` with the above settings.

 * How many total and activated parameters does it have? Activated means used by any given token. 
 * What batch size is needed to become FLOPs bound on TPU v5e?
 * How large are its KV caches per token?
 * How many FLOPs are involved in a forward pass with T tokens?
 
 Click here for the answer. (1) As an MoE, each MLP block now has $3 * E * D * F$ parameters, an increase of $E$ over the dense variant. Thus it now has $L * D * (3EF + 2H * (N + K)) + D * V$ or `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` total parameters, an increase of about 12x. For activated parameters, we have $k$ rather than $E$ activated parameters, for a total of `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`, an increase of less than 2x over the dense variant.

 (2) Because we have $E$ times more parameters for only $k$ times more FLOPs, our HBM roofline increases by a factor of $E/k$. That means on a TPU v5e we need about `240 * (16 / 2) = 1920` tokens.

 (3) The KV 缓存 size stays the same as the MoE character doesn’t change anything about the 注意力 mechanism.

 (4) This is still $2 \cdot \text{activated params} \cdot T$. Thus this is $2 * \text{31.2e9} * T$.

  Question 6: With MoEs, we can do “expert 分片”, where we split our experts across one axis of our mesh. In our standard notation, our first FFW weight has shape `[E, D, F]` and we shard it as [EZ, DX, FY] where `X` is only used during 训练 as our FSDP dimension. Let’s say we want to do 推理 on a TPU v5e:

 * What’s the HBM weight loading time for the above 模型 on a TPU v5e 8x16 slice with Y=8, Z=16? How much free HBM is available per TPU?
 * What is the smallest slice we could fit our 模型 on?
 
 Question 7 [2D 模型 分片]: Here we’ll work through the math of what the [ESTI paper](https://arxiv.org/pdf/2211.05102) calls 2D weight-stationary 分片. We describe this briefly in Appendix B, but try doing this problem first to see if you can work out the math. The basic idea of 2D weight stationary 分片 is to shard our weights along both the $D$ and $F$ axes so that each chunk is roughly square. This reduces the comms load and allows us to scale slightly farther.

 Here’s the 算法 for 2D weight stationary:

 * In[B, DX] = AllGatherYZ(In[B, DXYZ])
 * Tmp[B, FYZ] {UX} = In[B, DX] *D Win[DX, FYZ]
 * Tmp[B, FYZ] = AllReduceX(Tmp[B, FYZ] {UX})
 * Out[B, DX] {UYZ} = Tmp[B, FYZ] *F Wout[FYZ, DX]
 * Out[B, DXYZ] = ReduceScatterYZ(Out[B, DX] {UYZ})
 
  Your goal is to work out $T_\text{math}$ and $T_\text{comms}$ for this 算法 and find when it will outperform traditional 3D 模型 分片?

 Click here for the answer! Let’s work out $T_\text{math}$ and $T_\text{comms}$. All our FLOPs are fully sharded so as before we have $T_\text{math} = 4BDF / (N \cdot C)$ but our comms are now

 \[\begin{align*} T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} \end{align*}\] where we note that the AllReduce is twice as expensive and we scale our comms by the number of axes over which each 操作 is performed. Assuming we have freedom to choose our topology and assuming $F=4D$ (as in LLaMA-2), we claim (by some basic calculus) that the optimal values for $X$, $Y$, and $Z$ are $X = \sqrt{N / 8}$, $YZ = \sqrt{8N}$ so the total communication is

 \[T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}\] Firstly, copying from above, normal 1D 模型 parallelism would have $T_\text{模型 并行 comms} = 4BD / (3 \cdot W_\text{ici})$, so when are the new comms smaller? We have

 \[\begin{align*} T_\text{模型 并行 comms} &gt; T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} &gt; \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\ \iff N &gt; 128 \cdot \left(\frac{3}{4}\right)^2 = 81 \end{align*}\] For a general $F$, we claim this condition is

 \[N &gt; 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2\] So that tells us if we have more than 81 chips, we’re better off using this new scheme. Now this is a slightly weird result because we’ve historically found ourselves ICI bound at around ~20 way 张量 parallelism. But here, even if we’re communication-bound, our total communication continues to decrease with the number of total chips! What this tells us is that we can continue to increase our chips, increase our batch size, do more 参数 scaling, and see reduced 延迟.

  ### That’s all for Part 7! For Part 8, with a look at how we might serve LLaMA 3 on TPUs, click [here](../applied-推理). ## Appendix ### Appendix A: How real is the batch size &gt; 240 rule? The simple rule we provided above, that our batch size must be greater than 240 tokens to be compute-bound, is roughly true but ignores some ability of the TPU to prefetch the weights while other operations are not using all available HBM, like when doing inter-device communication.

 Here’s an empirical plot of 层 time (in microseconds) for a small Transformer with dmodel 8192, dff 32768, and only 2 matmuls per 层. This comes from [this Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing). You’ll see that step time increases very slowly up until around batch 240, and then increases linearly.

       Here’s the actual 吞吐量 in tokens / us. This makes the argument fairly clearly. Since our 层 is about 600M parameters sharded 4 ways here, we’d expect a 延迟 of roughly 365us at minimum.

       So at least in this 模型, we do in fact see 吞吐量 increase until about BS240 per data 并行 shard.

 ### Appendix B: 2D Weight Stationary 分片 As the topology grows, if we have access to higher dimensional meshes (like that of TPUs) it is possible to refine this further with “2D Weight 分片” by introducing a second 分片 axis. We call this “2D Weight Stationary”, and was described in more detail in the [Efficiently Scaling Transformer 推理 paper](https://arxiv.org/abs/2211.05102).

 Because we’re only 分片 the hidden \(F\) dimension in Megatron, it can become significantly smaller than \(E\) (the \(d_\text{模型}\) dimension) once the number of chips grows large with 1D 分片. This means at larger batch sizes, it can be more economical to perform a portion of the collectives over the hidden dimension after the first 层 of the MLP is applied.

       This figure shows:

 * 1D weight-stationary 分片, a.k.a. Pure Megatron 分片, where activations are fully replicated after AllGather, and weights are fully sharded over the hidden F dimension.
 * 2D weight stationary 分片, where weights are sharded over both the hidden F and reduction E dimension, and activations are sharded over the E dimension. We perform an AllGather on the (yz) axis before the first 层, then ReduceScatter on the (x) axis.
 
 For the 注意力 层, Megatron style 分片 is also relatively simple for smaller numbers of chips. However, Megatron happens over the \(n_\text{heads}\) dimension, which puts a limit on the amount of 分片 that is possible. Modifying the 2D 分片 for 注意力 (instead of 分片 the hidden dimension, we shard the \(n_\text{heads}\) dimension), we gain the ability to scale further.

 ### Appendix C: 延迟 bound communications As a recap, in [Section 3](../分片) we derived the amount of time it takes to perform an AllGather into a 张量 of size B on each TPU, over X chips on a 1D ring with links of full-duplex 带宽 of WICI and 延迟 Tmin.

 \[T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)\] For large B, the wall clock stays relatively constant because as you add more chips to the system, you simultaneously scale the amount of data movement necessary to perform the 操作 and the total 带宽 available.

       Because of the relatively low amounts of data being moved during 延迟 optimized 推理, collectives on activations are often bound by the 延迟 term (especially for small batch sizes). One can visualise the 延迟 quite easily, by counting the number of hops we need to complete before it is completed.

 On TPUs, if the 张量 size-dependent part of communication is less than 1 microsecond per hop (a hop is communication between two adjacent devices) we can be bottlenecked by the fixed overhead of actually dispatching the collective. With `4.5e10` unidirectional ICI 带宽, ICI communication becomes 延迟 bound when: \((\text{bytes} / n_\text{shards}) / 4.5e10 &lt; 1e-6\). For 8-way Megatron 分片, this is when `buffer_size &lt; 360kB`. This actually is not that small during 推理: with `BS=16` and `D=8192` in int8, our activations will use `16*8192=131kB`, so we’re already 延迟 bound.

 Takeaway: our comms become 延迟 bound when \(\text{total bytes} &lt; W_{ICI} \times 1e-6\). For instance, with 模型 parallelism over \(Y\), we become bound in int8 when \(Y &gt; BD / 45,000\).

 There’s a 并行 to be drawn here with the compute roofline — we are incurring the fixed cost of some small operations (延迟 for comms, 内存 带宽 for matmuls).

 ### Appendix D: Speculative Sampling When we really care about end to end 延迟, there is one extra trick we can employ called speculative sampling. As a recap, we usually generate tokens from a large Transformer one by one:

       With speculative sampling, we use a smaller, cheaper 模型 to generate tokens and then check the result with the big 模型. This is easiest to understand with greedy decoding:

 * We sample greedily from some smaller, cheaper 模型. Ideally we use a 模型 trained to match the larger 模型, e.g. by distillation, but it could be as simple as simply using n-grams or token matching a small corpus of text.
 * After we’ve generated K tokens, we use the big 模型 to compute the next-token logits for all the tokens we’ve generated so far.
 * Since we’re decoding greedily, we can just check if the token generated by the smaller 模型 has the highest probability of all possible tokens. If one of the tokens is wrong, we take the longest correct prefix and replace the first wrong token with the correct token, then go back to (1). If all the tokens are correct, we can use the last correct logit to sample an extra token before going back to (1).
 
 Why is this a 延迟 win? This scheme still requires us to do the FLOPs-equivalent of one forward pass through the big 模型 for every token, but because we can batch a bunch of tokens together, we can do all these FLOPs in one forward pass and take advantage of the fact that we’re not compute-bound to score more tokens for free.

 Every accepted token becomes more expensive in terms of FLOPs on average (since some will be rejected, and we have to call a draft 模型), but we wring more FLOPs out of the hardware, and the small 模型 is cheap, so we win overall. We also share KV 缓存 loads across multiple steps, so speculative decoding can also be a 吞吐量 win for long context. Since everything has been checked by the big 模型, we don’t change the sampling distribution at all (though the exact trajectory will differ for non-greedy).

 Traditionally, speculative decoding relies on the existence of a smaller 模型 with a similar sampling distribution to the target 模型, e.g. LLaMA-2 2B for LLaMA-2 70B, which often doesn’t exist. Even when this is available, the smaller drafter can still be too expensive if the acceptance rate is low. Instead, it can be helpful to embed a drafter within the main 模型, for instance by adding a dedicated drafter head to one of the later layers of the base 模型. Because this head shares most of its parameters with the main 模型, it’s faster to run and matches the sampling distribution more closely.

 For normal autoregressive sampling the token/s is the same as the step time. We are still beholden to the theoretical minimum step time according to the Arithmetic Intensity section here (in fact, Speculative Sampling step times are usually quite a bit slower than normal autoregressive sampling, but because we get more than 1 token out per step on average we can get much better tokens/s).

      Figure: this figure shows the per-step 延迟 and speculation success rate for Chinchilla (a 70B 模型 from DeepMind) with a 4B 参数 drafter (small 模型). For XSum (a natural language dataset), the ideal amount of speculation is about 3-4 tokens ahead, while HumanEval (a coding dataset) is more predictable and sees wins from more aggressive speculation.  How does this work for non-greedy decoding? This is a bit more complicated, but essentially boils down to a Metropolis-Hastings inspired 算法 where we have \(P_{\text{draft 模型}}(\text{chosen token})\) and \(P_{\text{target 模型}}(\text{chosen token})\) derived from the logits, and reject the chosen token probabilistically if the ratio of these probabilities is smaller than some threshold.

 These [two](https://arxiv.org/abs/2211.17192) [papers](https://arxiv.org/abs/2302.01318) derived this concurrently and have good examples of how this works in practice.

 Takeaway: Speculative sampling is yet another powerful lever for trading 吞吐量 for better per token 延迟. However, in the scenario where batch size is limited (e.g. small hardware footprint or large KV caches), it becomes a win-win.

      ### Miscellaneous *Work done at Google DeepMind, now at MatX.

   ### Citation For attribution in academic contexts, please cite this work as:

  ```
    Austin et al., "How to Scale Your 模型", Google DeepMind, online, 2025.

```  or as a BibTeX entry:

  ```
    @article{scaling-book,
      title = {How to Scale Your 模型},
      author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
      and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
      publisher = {Google DeepMind},
      howpublished = {Online},
      note = {Retrieved from https://jax-ml.github.io/scaling-book/},
      year = {2025}
    }

```       Please enable JavaScript to view the [comments powered by giscus.](http://giscus.app/?ref_noscript)      © Copyright 2026 . Powered by [Jekyll](https://jekyllrb.com/) with [al-folio](https://github.com/alshedivat/al-folio) theme. Hosted by [GitHub Pages](https://pages.github.com/).

---

## 🔗 相关资源

1. **官方文档**：
   - [JAX官方文档](https://jax.readthedocs.io/)
   - [XLA编译优化](https://www.tensorflow.org/xla)
   - [TPU技术指南](https://cloud.google.com/tpu/docs)

2. **参考论文**：
   - [Transformer原始论文](https://arxiv.org/abs/1706.03762)
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

3. **实践项目**：
   - [JAX示例代码库](https://github.com/google/jax)
   - [Transformer实现示例](https://github.com/huggingface/transformers)
   - [TPU使用教程](https://github.com/tensorflow/tpu)

## 💡 学习建议

### 理论学习
1. 先通读全文，了解整体框架
2. 重点理解核心概念和技术原理
3. 结合图表和公式深入理解

### 实践学习
1. 运行教材中的代码示例
2. 尝试修改参数观察效果
3. 应用到自己的项目中

### 深入学习
1. 阅读参考文献和扩展阅读
2. 参与相关技术社区讨论
3. 关注最新的技术发展

---

*本翻译由OpenClaw自动生成，正在不断完善中。*
*翻译问题反馈：请通过博客反馈渠道联系。*
