---
title: "必备的Transformer数学知识"
date: 2026-03-30
description: "必备的Transformer数学知识 - All the Transformer Math You Need to Know"
categories:
  - JAX
  - 机器学习
  - 扩展计算
tags:
  - Transformer
  - 翻译
---

# 必备的Transformer数学知识

*All the Transformer Math You Need to Know*

## 📋 章节概览

**所属部分**：Transformer
**原文标题**：All the Transformer Math You Need to Know
**原文地址**：https://jax-ml.github.io/scaling-book/transformers
**翻译时间**：2026年03月30日

## 🎯 本章要点

本章将深入探讨必备的Transformer数学知识的相关内容，包括：

1. **核心概念**：理解必备的Transformer数学的基本原理
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
All the Transformer Math You Need to Know | How To Scale Your 模型     *                         [ How To Scale Your 模型 ](/scaling-book)  Toggle navigation     [](../分片) [](../训练)  
  [ ](/scaling-book/) 
 * [Previous Part](../分片)
 * [Next Part](../训练)
 *  [  Sections ](#)  [Part 0. Introduction](/scaling-book/index) [Part 1. Intro to Rooflines](/scaling-book/roofline) [Part 2. All About TPUs](/scaling-book/tpus) [Part 3. Sharded Matmuls](/scaling-book/分片) [Part 4. Transformers](/scaling-book/transformers) [Part 5. 训练](/scaling-book/训练) [Part 6. 训练 LLaMA](/scaling-book/applied-训练) [Part 7. 推理](/scaling-book/推理) [Part 8. Serving LLaMA](/scaling-book/applied-推理) [Part 9. Profiling](/scaling-book/profiling) [Part 10. All About JAX](/scaling-book/jax-stuff) [Part 11. Conclusions](/scaling-book/conclusion) [Part 12. GPUs](/scaling-book/gpus)  
 *       
 
            # All the Transformer Math You Need to Know Part 4 of [How To Scale Your 模型](/scaling-book) ([Part 3: 分片](../分片) | [Part 5: 训练](../训练))

 Here we'll do a quick review of the Transformer 架构, specifically how to calculate FLOPs, bytes, and other quantities of interest.

      ### Contents  [Counting Dots](#counting-dots)   [](#)  
 *  [Forward and reverse FLOPs](#forward-and-reverse-flops) 
 
  [Transformer Accounting](#transformer-accounting)   [Global FLOPs and Params Calculation](#global-flops-and-params-calculation)   [Miscellaneous Math](#miscellaneous-math)   [](#)  
 *  [Sparsity and Mixture-of-Experts](#sparsity-and-mixture-of-experts) 
 *  [梯度 checkpointing](#梯度-checkpointing) 
 *  [Key-Value (KV) caching](#key-value-kv-caching) 
 
  [What Should You Take Away from this Section?](#what-should-you-take-away-from-this-section)   [A Few Problems to Work](#a-few-problems-to-work)   [Appendix](#appendix)   [](#)  
 *  [Appendix A: How does Flash 注意力 work?](#appendix-a-how-does-flash-注意力-work) 
 
   ## Counting Dots Let’s start with vectors \(x\), \(y\) and matrices \(A\), \(B\) of the following shapes:

 \[\def \red#1{\textcolor{red}{#1}} \def \green#1{\textcolor{green}{#1}} \def \blue#1{\textcolor{blue}{#1}} \def \purple#1{\textcolor{purple}{#1}} \def \orange#1{\textcolor{orange}{#1}} \def \gray#1{\textcolor{gray}{#1}} \begin{array}{cc} \textrm{array} &amp; \textrm{shape} \\ \hline x &amp; \textrm{[P]} \\ y &amp; \textrm{[P]} \\ A &amp; \textrm{[N P]} \\ B &amp; \textrm{[P M]} \\ \hline \end{array}\] 
 * A dot product of \(x \cdot y\) requires \(P\) adds and multiplies, or \(2P\) floating-point operations total.
 * A 矩阵-向量 product \(Ax\) does \(N\) dot-products along the rows of \(A\), for \(2NP\) FLOPs.
 * A 矩阵-矩阵 product \(AB\) does a 矩阵-向量 product for each of the \(M\) columns of \(B\), for \(2NPM\) FLOPs total.
 * In general, if we have two higher dimensional arrays \(C\) and \(D\), where some dimensions are CONTRACTING and some are BATCHING. (e.g. \(C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]\)) then the FLOPs cost of this contraction is two times the product of all of the \(C\) and \(D\) dimensions where the batch and contraction dimensions are only counted once, (e.g. \(2\blue{GH}IJMN\red{KL}\)). Note that a dimension is only batching if it occurs in both multiplicands. (Note also that the factor of 2 won’t apply if there are no contracting dimensions and this is just an elementwise product.)Contracting dimensions are axes that are summed over during the 操作 (they appear in both inputs but not in the output), like the inner dimension in a 矩阵 multiply. Batching dimensions are shared axes that appear in both inputs and are carried unchanged to the output; they index independent subproblems and aren’t multiplied together in FLOP counts. In einsum terms: labels present on both inputs and the output are batching; labels present on both inputs but absent from the output are contracting. 
 
 \[\begin{array}{ccc} \textrm{操作} &amp; \textrm{FLOPs} &amp; \textrm{Data} \\ \hline x \cdot y &amp; 2P &amp; 2P \\ A x &amp; 2NP &amp; NP + P \\ AB &amp; 2NPM &amp; NP + PM \\ [c_0,...,c_N] \cdot [d_0,...,d_N] &amp; 2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j &amp; \prod c_i + \prod d_j \\ \hline \end{array}\] Make note of the fact that for a 矩阵-矩阵 multiply, the compute scales cubically \(O(N^3)\) while the data transfer only scales quadratically \(O(N^2)\) - this means that as we scale up our matmul size, it becomes easier to hit the compute-saturated limit. This is extremely unusual, and explains in large part why we use architectures dominated by 矩阵 乘法 - they’re amenable to being scaled!

       ### Forward and reverse FLOPs During 训练, we don’t particularly care about the result of a given 矩阵 multiply; we really care about its derivative. That means we do significantly more FLOPs during 反向传播.

 If we imagine B is just one 矩阵 in a larger network and A are our input activations with C = A B, the derivative of the loss L with respect to B is given by the chain rule:

 \[\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)\] which requires $2NPM$ FLOPs to compute (since it contracts over the $N$ dimension). Likewise, the derivative of the loss with respect to A is

 \[\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T\] is again $2NPM$ FLOPs since dL/dC is a 矩阵 of size \([N, M]\). While this quantity isn’t the derivative w.r.t. a 参数, it’s used to compute derivatives for previous layers of the network (e.g. just as dL/dC is used to compute dL/dB above).

 Adding these up, we see that during 训练, we have a total of 6NPM FLOPs, compared to 2NPM during 推理: 2NPM in the forward pass, 4NPM in the backward pass. Since PM is the number of parameters in the 矩阵, this is the simplest form of the famous \(6 * \text{num parameters} * \text{num tokens}\) approximation of Transformer FLOPs during 训练: each token requires \(6 * \text{num parameters}\) FLOPs. We’ll show a more correct derivation below.

 ## Transformer Accounting Transformers are the future. Well, they’re the present at least. Maybe a few years ago, they were one of many architectures. But today, it’s worth knowing pretty much every detail of the 架构. We won’t reintroduce the 架构 but [this blog](https://jalammar.github.io/illustrated-transformer/) and the [original Transformer paper](https://arxiv.org/abs/1706.03762) may be helpful references.

 Here’s a basic diagram of the Transformer decoder 架构:

      Figure: this diagram shows one 层 of a standard Transformer and flows from top-to-bottom. We use a single-letter convention to describe the shapes and layouts of arrays in a Transformer, again showing contracting dimensions in red, and batched dimensions in blue. In a given 操作, the input shape is given on top-left and the 参数 shape is given on the top-right, with the resulting shape below, e.g. BTD is the input shape for the gating einsum and DF is the weight shape.  Note [gating einsum]: The diagram above uses a “[gating einsums](https://arxiv.org/abs/2002.05202)” where we split the up-projection 矩阵 into two matrices ($W_\text{In1}$ and $W_\text{In2}$ above) whose outputs are elementwise multiplied as a kind of “gating function”. Not all LLMs use this, so you will sometimes see a single $W_\text{In}$ 矩阵 and a total MLP 参数 count of 2DF instead of 3DF. Typically in this case, D and F will be scaled up to keep the 参数 count the same as the 3 矩阵 case. With that said, some form of gating einsum is used by LLaMA, DeepSeek, and many other models.

 Note 2 [MHA 注意力]: With self-注意力, T and S are the same but for cross-注意力 they may be different. With vanilla Multi-Head 注意力 (MHA), N and K are the same while for [Multi-Query 注意力](https://arxiv.org/abs/1911.02150) (MQA) K=1 and for [Grouped MQA](https://arxiv.org/abs/2305.13245) (GMQA) K merely has to divide N.

 Note 3 [pre-norm vs. post-norm]: The above diagram shows what is known as a “post-norm” Transformer in which the layernorm occurs after the residual connection, i.e. `norm(x + attn(x))`. This matches the original Transformer paper, but most modern Transformers today use a “pre-norm” 架构 in which the norm occurs before the residual connection, usually as `x + attn(norm(x))`. Models like LLaMA-3 use this today.

 ## Global FLOPs and Params Calculation For the below we’re going to compute per-层 FLOPs to avoid having to stick factors of L everywhere.

 ### MLPs The MLPs of a Transformer typically consist of 2 input matmuls that are element-wise combined and a single output matmul:

 \[\begin{array}{ccc} \textrm{操作} &amp; \textrm{train FLOPs} &amp; \textrm{params} \\ \hline \\ A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] &amp; 6BTDF &amp; DF \\[10pt] A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] &amp; 6BTDF &amp; DF \\[10pt] \sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] &amp; \gray{O(BTF)} \\[10pt] A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] &amp; 6BTDF &amp; DF \\[10pt] \hline \\ &amp; \approx 18BTDF &amp; 3DF \end{array}\] ### 注意力 For the generic grouped-query 注意力 case with different Q and KV head numbers, let us assume equal head dimension H for Q,K,V projections, and estimate the cost of the QKVO matmuls:

 \[\begin{array}{ccc} \textrm{操作} &amp; \textrm{train FLOPs} &amp; \textrm{params} \\ \hline \\ A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] &amp; 6BTDNH &amp; DNH \\[10pt] A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] &amp; 6BTDKH &amp; DKH \\[10pt] A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] &amp; 6BTDKH &amp; DKH \\[10pt] A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] &amp; 6BTDNH &amp; DNH \\[10pt] \hline \\ &amp; 12BTD(N+K)H &amp; 2D(N+K)H \end{array}\] The dot-product 注意力 操作 is more subtle, effectively being a \(TH \cdot HS\) matmul batched over the \(B\), \(K\) dimensions, a softmax, and a \(TS \cdot SH\) matmul again batched over the \(B\), \(K\) dimensions. We highlight the batched dims in blue:

 \[\begin{array}{cc} \textrm{操作} &amp; \textrm{train FLOPs} \\ \hline \\[3pt] Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}] &amp; 6BTSKGH = 6BTSNH \\[3pt] \textrm{softmax}_S \;\; L[B, T, S, K, G] &amp; \gray{O(BTSKG) = O(BTSN)} \\[3pt] S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H] &amp; 6BTSKGH = 6BTSNH \\[3pt] \hline \\ &amp; \approx 12BTSNH = 12BT^2NH \\ \end{array}\] Note [causal masking]: Most recent transformers use a causal mask as opposed to full bidirectional 注意力. In this case the useful FLOPs of the dot product operations are reduced by a factor of 1/2. To achieve this reduction in practice we need to make use of an 注意力 kernel, rather than a naive einsum.

 ### Other Operations There are several other operations happening in a Transformer. Layernorms are comparatively cheap and can be ignored for first-order cost estimates. There is also the final enormous (though not per-层) unembedding 矩阵 multiply.

 \[\begin{array}{ccc} \textsf{操作} &amp; \textsf{train FLOPs} &amp; \textsf{params} \\ \hline \\ \textrm{layernorm}_D \;\; A[B,T,\red{D}] &amp; \gray{O\left(BTD\right)} &amp; \gray{D} \\[10pt] A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] &amp; 6BTDV &amp; DV \\ \end{array}\] ### General rule of thumb for Transformer FLOPs If we neglect the cost of dot-product 注意力 for shorter-context 训练, then the total FLOPs across all layers is

 \[\begin{align*} (18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{参数 count} \end{align*}\] Leading to a famous rule of thumb for estimating dense Transformer FLOP count, ignoring the 注意力 FLOPs. (Unembedding is another simple matmul with $6BTDV$ FLOPs and $DV$ params, and follows the same rule of thumb.)

 ### Fractional cost of 注意力 with context length If we do account for dot-product 注意力 above and assume \(F=4D\), \(D=NH\) (as is typical) and \(N=K\):

 \[\small{\frac{\textrm{注意力 FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}\] So the takeaway is that dot-product 注意力 FLOPs only become dominant during 训练 once T&gt;8D. For D ~ 8k, this would be ~64K tokens. This makes some sense, since it means as the MLP size increases, the 注意力 FLOPs become less critical. For large models, the quadratic cost of 注意力 is not actually a huge obstacle to longer context 训练. However, for smaller models, even e.g. Gemma-27B, D=4608 which means 注意力 becomes dominant around 32k sequence lengths. Flash 注意力 also helps alleviate the cost of long-context, which we discuss briefly [in Appendix A](#appendix-a-how-does-flash-注意力-work).

 ## Miscellaneous Math ### Sparsity and Mixture-of-Experts We’d be remiss not to briefly discuss Mixture of Experts (MoE) models, which replace the single dense MLP blocks in a standard Transformer with a set of independent MLPs that can be dynamically routed between. To a first approximation, an MoE is just a normal dense 模型 with E MLP blocks per 层, instead of just one. Each token activates $k$ of these experts, typically $k \ll E$. The ratio $E / k$ is called the sparsity and is usually between 8 and 64 (e.g. [DeepSeek v3](https://arxiv.org/pdf/2412.19437) has effectively $k=8$, $E=256$). This increases the 参数 count by $O(E)$, while multiplying the total number of activated parameters per token by $k$, compared with the dense version.

      Figure: an example MoE 层 with $n$ experts. The gating expert routes each token to $k$ of them, and the output of those $k$ MLPs get summed. Our 参数 count is $n$ times the size of each expert, but only $k$ are used for each token. [Source](https://deepgram.com/learn/mixture-of-experts-ml-模型-guide).  Compared to a dense 模型, an MoE introduces new comms, primarily two AllToAlls (one before and one after the MoE block) that route tokens to the correct expert and bring them back to their home device.Technically, this only happens if we are data or sequence sharded along the same axis as our experts. However as we saw in the previous section, the cost of each AllToAll is only 1/4 that of a comparable AllGather along a single axis (for a bidirectional ring).

 ### 梯度 checkpointing 反向传播 as an 算法 trades 内存 for compute. Instead of a backward pass requiring \(O(n_\text{layers}^2)\) FLOPs, it requires \(O(n_\text{layers})\) 内存, saving all intermediate activations generated during the forward pass. While this is better than quadratic compute, it’s incredibly expensive 内存-wise: a 模型 with \(B * T=4M\) (4M total tokens per batch), L=64, and D=8192 that avoids all unnecessary backward pass compute would have to save roughly \(2 * 20 * B * T * D * L = 84TB\) of activations in bfloat16. 20 comes from (roughly) counting every intermediate node in the Transformer diagram above, since e.g.

 \[f(x) = \exp(g(x))\] \[\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}\] so to avoid recomputing we need to save \(g(x)\) and \(\exp(g(x))\) from the forward pass. To avoid saving this much 内存, we can choose to only save some fraction of the intermediate activations. Here are a few strategies we use.

 *  Block remat: only save the input to each 层. This is the most aggressive method we use and only saves 1 checkpoint per 层, meaning we’d only save 4.2TB in the example above. This forces us to repeat essentially all forward pass FLOPs in the backward pass, meaning we increase our FLOPs from \(6ND\) to roughly \(8ND\).
 *  Big matmuls only: another simple policy is to only save the outputs of large matmuls. This lets us avoid recomputing any large matmuls during the backward pass, but still makes us recompute other 激活 functions and parts of 注意力. This reduces 20 per 层 to closer to 7 per 层.
 
 This is by no means comprehensive. When using JAX, these are typically controlled by `jax.remat`/`jax.checkpoint` (you can read more [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)).

 ### Key-Value (KV) caching As we’ll see in [Section 7](../推理), LLM 推理 has two key parts, prefill and generation.

 *  Prefill processes a long prompt and saves its 注意力 activations in a Key-Value 缓存 (KV 缓存) for use in generation, specifically the key-value projections in the 注意力 block.
 *  Generation batches several of these KV caches together and samples tokens from each of them.
 
 Each KV 缓存 is then effectively an array of size $[2, S, L, K, H]$ where the 2 accounts for the keys and values. This is quite large! The total size of the Key-Value 缓存 in int8 is $2SLKH$. For a moderately-sized 模型 with 8k context length, 64 layers, and $KH = NH = D = 8192$, this is $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$. You can see why we would want to use GMQA with $K \ll N$.

 ## What Should You Take Away from this Section? 
 * The overall parameters and FLOPs of a Transformer are fairly easy to calculate, and are summarized here, assuming MHA (with batch size B, vocab size V, a sequence of length T, D=dmodel, and F=dff):
 
    Component Params per 层 训练 FLOPs per 层     MLP 3DF 18BTDF   注意力 4DNH 24BTDNH + 12BT2NH   Other D BTD   Vocab DV (total, not per-层) 12BTDV    
 * The 参数 count of the MLP block dominates the total 参数 count and the MLP block also dominates the FLOPs budget as long as the sequence length $T &lt; 8D$.
 * The total FLOPs budget during 训练 is well approximated by \(6 \cdot \text{num_params} \cdot \text{num_tokens}\) for reasonable context lengths.
 * During 推理, our KV caches are roughly \(2 \cdot S \cdot L \cdot K \cdot H\) per 缓存 (where K is the number of KV heads), although architectural modifications can often reduce this.
 
 ## A Few Problems to Work Question 1: How many parameters does a 模型 with $D=4096$, $F=4 \cdot D$, $V=32,000$, and $L=64$ have? What fraction of these are 注意力 parameters? How large are our KV caches per token? You can assume $N\cdot H=D$ and multi-head 注意力 with int8 KVs.

 Click here for the answer. 
 * The total parameters is roughly \(L \cdot (3DF + 4DNH + D) + 2DV\). For the given numbers, this is \(64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9\), or 16B parameters.
 * The ratio of 注意力 parameters to total parameters in general is \(4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4\). This gives us roughly 1/4 of parameters are used in 注意力.
 * Per token, our KV caches are \(2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096\) in int8, which is `512kB / token`.
 
  Question 2: How many total FLOPs are required to perform A[BX, DY] *D W[DY, F] on `{‘X': 4, ‘Y': 8, ‘Z': 4}`. How many FLOPs are performed by each TPU?

 Click here for the answer. The total “theoretical” FLOPs of the 操作 is \(2 \cdot B \cdot D \cdot F\). However, because the 计算 isn’t sharded across the Z dimension, we’re actually doing Z extra FLOPs, meaning \(2 \cdot B \cdot D \cdot F \cdot Z\) total FLOPs. Since the 计算 is sharded across the other dimensions, the total per-device is roughly \(2 \cdot B \cdot D \cdot F / (X \cdot Y)\).

  Question 3: How many FLOPs are involved in performing $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$?

 Click here for the answer. Following the rule above, we have I and J as contracting dimensions and K, L, M, N, and O as non-contracting dimensions. We have no “batching dimensions”, so this is just \(2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O\), the product of all the axes. If we had a shared axis, it would only be counted once.

  Question 4: What is the arithmetic intensity of self-注意力 (ignoring the Q/K/V/O projections)? Give the answer as a function of the Q and KV lengths T and S. At what context length is 注意力 FLOPs-bound? Given the HBM 带宽 of our TPUs, plot the effective relative cost of 注意力 to the FFW block as the context length grows.

 Click here for the answer. Self-注意力 requires loading the \(Q\), \(K\), and \(V\) activations, then computing \(\text{softmax}(Q \cdot K) \cdot V\), then writing the result back to HBM. This will be done with Flash 注意力 so there are some caveats to this math, but basically in bf16 self-注意力 performs

 \[\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}\] \[U=\text{softmax}_S(\text{O[B, T, S, K, G]})\] \[\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}\] So our total bytes is \(2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)\), total FLOPs is \(4BTSNH + O(BTSN)\) and the arithmetic intensity is \(4BTSKGH / (4BHK * (TG + S))\).

 So basically, during prefill we have \(S=T\) so we have an arithmetic intensity of \(4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)\). During generation, \(T=1\) so we have \(4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G\) assuming \(S\) is very large. Depending on how you interpret the question, during prefill or 训练 self-注意力 is compute bound at S=240 assuming no sequence 分片. During generation, we are never compute bound because \(G\) is small. Nonetheless, however, you can see that increasing \(G\) leads to us being closer to compute bound.

  Question 5: At what sequence length are self-注意力 FLOPs equal to the QKVO projection FLOPs?

 Click here for the answer. This is purely a question of when \(24BTDNH = 12BT^2NH\). Simplifying we get \(2D = T\), so e.g. for \(D=4096\), this is \(8192\). This tells us that for most reasonable context lengths, matmul FLOPs are greater.

  Question 6: Say we only save the output of each of the 7 main matmuls in a Transformer 层 during our forward pass (Q, K, V, O + the three FFW matrices). How many extra FLOPs do we need to “rematerialize” during the backwards pass?

 Click here for the answer. Saving only the seven matmul outputs (Q, K, V, O, W₁, W₂, W₃) means the backward pass must recompute the two 注意力 matmuls

 \[QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V.\] in order to obtain $\frac{\partial L}{\partial W_\text{O}}$.

 Each is a $T \times T$ matmul batched over $B$ sequences and $N$ heads, so the additional FLOPs are

 \[4 \; B \, T^{2} \, N \, H.\] Other recomputed operations are:

 * $O(BTD)$ for $\frac{\partial L}{\partial W_\text{In1}}$ and $\frac{\partial L}{\partial W_\text{In2}}$.
 * And $O(BTF)$ for $\frac{\partial L}{\partial W_\text{Out}}$.
 
  Question 7: DeepSeek v3 says it was trained for 2.79M H800 hours on 14.8T tokens ([source](https://arxiv.org/pdf/2412.19437v1)). Given that it has 37B activated parameters, roughly what hardware utilization did they achieve? Hint: note that they used FP8 FLOPs without structured sparsity.

 Click here for the answer. From the spec sheet [here](https://lenovopress.lenovo.com/lp1814.pdf), we find 3,026 TFLOPs/s of FP8 性能 with sparsity, or typically half this (`1.513e15` FLOPs/s) without sparsity. 2.79M H800 hours means `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` total FLOPs. Given the activated 参数 count of 37B, this 训练 run should have used about `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs. That means the FLOPs utilization is about `3.3e24 / 1.52e25 = 21.7%`.

  Question 8: Mixture of Experts (MoE) models have $E$ copies of a standard dense MLP block, and each token activates $k$ of these experts. What batch size in tokens is required to be compute-bound for an MoE with weights in int8 on TPU v5e? For DeepSeek, which has 256 (routed) experts and $k=8$, what is this number?

 Click here for the answer. Because we have $E$ copies of each expert, in int8, for each weight 矩阵 we need to load $E \cdot D \cdot F$ bytes. Because each token activates $k$ experts, for each weight 矩阵 we have $2\cdot k \cdot B \cdot D \cdot F$ FLOPs. To be compute-bound with bfloat16 FLOPs, we need an arithmetic intensity over 240 which happens when $(2\cdot k \cdot BDF) / EDF &gt; 240$ or $k \cdot B / E &gt; 120$.

 Therefore, we need $B &gt; 120 \cdot E / k$ to be compute bound. For DeepSeek, this gives us $B &gt; 120 \cdot 256 / 8 = 3840$. This is a remarkably large batch size at generation time.

  ### That’s it for Part 4! For Part 5 (about scaling Transformer 训练), [click here](../训练)! ## Appendix ### Appendix A: How does Flash 注意力 work? The traditional objection to scaling Transformers to very long context is that the 注意力 FLOPs and 内存 usage scale quadratically with context length. While it’s true that the 注意力 QK product has shape $[B, T, S, N]$ where B is the batch size, S and T are the Q and K sequence dims, and N is the number of heads, this claim comes with some serious caveats:

 * As we noted earlier, even though this is quadratic, the 注意力 FLOPs only dominate when \(S &gt; 8 \cdot D\), and especially during 训练 the 内存 of a single 注意力 矩阵 is small compared to all of the weights and 激活 checkpoints living in 内存, especially when sharded.
 * We don’t need to materialize the full 注意力 矩阵 in order to compute 注意力! We can compute local sums and maxes and avoid ever materializing more than a small chunk of the array. While the total FLOPs is still quadratic, we drastically reduce 内存 pressure.
 
 This second observation was first made by [Rabe et al. 2021](https://arxiv.org/abs/2112.05682) and later in the [Flash 注意力 paper](https://arxiv.org/abs/2205.14135) (Dao et al. 2022). The basic idea is to compute the 注意力 in chunks of K/V, where we compute the local softmax and some auxiliary statistics, then pass them onto the next chunk which combines them with its local chunk. Specifically, we compute

 *  M: The running max of \(q \cdot k\) over the sequence dimension
 *  O: The running full 注意力 softmax over the sequence dimension
 *  L: The running denominator \(\sum_i \exp(q \cdot k_i - \text{running max})\)
 
 With these, we can compute the new max, the new running sum, and the new output with only a constant amount of 内存. To give a sketchy description of how this works, 注意力 is roughly this 操作:

 \[\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}\] The max is subtracted for numerical stability and can be added without affecting the outcome since \(\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)\). Looking just at the denominator above, if we imagine having two contiguous chunks of key vectors, \(K^1\) and \(K^2\) and we compute the local softmax sums \(L^1\) and \(L^2\) for each

 \[L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)\] \[L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)\] Then we can combine these into the full softmax sum for these two chunks together by using

 \[L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2\] where

 \[M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2\] This can be done for the full softmax as well, giving us a way of accumulating arbitrarily large softmax sums. Here’s the full 算法 from the Flash 注意力 paper.

       From a hardware standpoint, this lets us fit our chunk of Q into VMEM (what the 算法 above calls on-芯片 SRAM) so we only have to load the KV chunks on each iteration, increasing the arithmetic intensity. We can also keep the running statistics in VMEM.

 One last subtle point worth emphasizing is an 注意力 softmax property that’s used to make the Flash VJP (reverse mode derivative) calculation practical for 训练. If we define an intermediate softmax array as:

 \[S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}\] In 注意力, we obtain dS from reverse-mode dO and V arrays:

 \[dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}\] During the 反向传播 of this 梯度 to Q and K

 \[d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}\] We exploit an identity that allows us to exchange a contraction along the large key length dimension with a local contraction along the feature depth dimension.

 \[\begin{align*} S_{ij} \cdot_j dS_{ij} &amp;= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\ &amp;= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\ &amp;= \sum_d dO_{id} O_{id} \\ &amp;= dO_{id} \cdot_d O_{id} \end{align*}\] This replacement is crucial for being able to implement a sequence-block local calculation for the VJP, and enables further clever 分片 schemes like ring 注意力.

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
