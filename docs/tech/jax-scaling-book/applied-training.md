---
title: "在TPU上训练LLaMA 3"
date: 2026-03-30
description: "在TPU上训练LLaMA 3 - Training LLaMA 3 on TPUs"
categories:
  - JAX
  - 机器学习
  - 扩展计算
tags:
  - Transformer
  - 翻译
---

# 在TPU上训练LLaMA 3

*Training LLaMA 3 on TPUs*

## 📋 章节概览

**所属部分**：Transformer
**原文标题**：Training LLaMA 3 on TPUs
**原文地址**：https://jax-ml.github.io/scaling-book/applied-training
**翻译时间**：2026年03月30日

## 🎯 本章要点

本章将深入探讨在TPU上训练LLaMA 3的相关内容，包括：

1. **核心概念**：理解在TPU上训练LLaMA 3的基本原理
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
训练 LLaMA 3 on TPUs | How To Scale Your 模型     *                         [ How To Scale Your 模型 ](/scaling-book)  Toggle navigation     [](../训练) [](../推理)  
  [ ](/scaling-book/) 
 * [Previous Part](../训练)
 * [Next Part](../推理)
 *  [  Sections ](#)  [Part 0. Introduction](/scaling-book/index) [Part 1. Intro to Rooflines](/scaling-book/roofline) [Part 2. All About TPUs](/scaling-book/tpus) [Part 3. Sharded Matmuls](/scaling-book/分片) [Part 4. Transformers](/scaling-book/transformers) [Part 5. 训练](/scaling-book/训练) [Part 6. 训练 LLaMA](/scaling-book/applied-训练) [Part 7. 推理](/scaling-book/推理) [Part 8. Serving LLaMA](/scaling-book/applied-推理) [Part 9. Profiling](/scaling-book/profiling) [Part 10. All About JAX](/scaling-book/jax-stuff) [Part 11. Conclusions](/scaling-book/conclusion) [Part 12. GPUs](/scaling-book/gpus)  
 *       
 
            # 训练 LLaMA 3 on TPUs Part 6 of [How To Scale Your 模型](/scaling-book) ([Part 5: 训练](../训练) | [Part 7: 推理](../推理))

 Let's take a close look at how we'd train LLaMA 3 models on TPU v5p using what we've learned in the previous section. How big are they? How expensive is 训练 in different configurations? How are they sharded? Let's work through some back-of-the-envelope estimates for how the previous sections map onto real models.

      ### Contents  [What does LLaMA 3 look like?](#what-does-llama-3-look-like)   [Counting parameters and FLOPs](#counting-parameters-and-flops)   [How to shard LLaMA 3-70B for 训练](#how-to-shard-llama-3-70b-for-训练)   [Worked Problems](#worked-problems)    Our goal in this section is to apply results from the previous section to a very practical problem: 训练 the LLaMA 3 family (herd) of models. Unlike the previous sections we want you to do a lot of this work yourself. For this reason, we’ve hidden the answers to each section so you can try to answer it first. Try grabbing a pen and doing it by hand!

 ### What does LLaMA 3 look like? The LLaMA-3 模型 family includes 3 main models: LLaMA 3 8B, 70B, and 405B. We’ll mostly focus on 70B, and leave 8B and 405B for you to explore in the problem section at the end. Here’s the 架构 for LLaMA 3-70B, taken from the LLaMA [HuggingFace page](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json).

    hyperparam value     \(n_\text{layers}\) (L) 80   \(d_\text{模型}\) (D) 8,192   \(d_{ff}\) (F) 28,672   \(n_\text{heads}\) (N) 64   \(n_\text{kv_heads}\) (K) 8   \(d_\text{qkv}\) (H) 128   \(n_\text{embeddings}\) (V) 128,256    To highlight how easy this is to find, here’s the config itself, along with a mapping:

       It’s useful to make a big table with these numbers for many different open-source LLMs, so you can quickly compare the design decisions they’ve made.

 ### Counting parameters and FLOPs Question: From this table, can we calculate the LLaMA 3-70B 参数 count? 🤫 Let’s apply the content of [Section 4](../transformers) and see if we can get 70B!

    param formula count     FFW params d_model * d_ff * 3 (for SwiGLU gate, up, and down projections) * n_layers 8,192 * 8,192 * 3.5 * 3 * 80 = 56.3e9    Vocab params 2 (input and output embeddings) * n_embeddings * d_model 2 * 128,256 * 8,192 = 2.1e9    注意力 params n_layers * [ 2 (for q 嵌入 and concatenated output projection) * d_model * n_heads * d_qkv + 2 (for k and v) * d_model * n_kv_heads * d_qkv] 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = 12e9        56.3e9 + 2.1e9 + 12e9 = 70.4e9     That’s great! We get the number we expect. You’ll notice as expected that the FFW parameters totally dominate the overall 参数 count, although 注意力 is non-trivial.

 Takeaway: The 3 big weight matrices in the MLP block are so much larger than all the other arrays in the Transformer that we can typically almost ignore all other parameters when reasoning about 模型 内存 or FLOPs. For LLaMA 3-70B, they represent 56B of 70B parameters.

 Let’s look at FLOPs now! Remember the general rules for 训练 from [Section 4](../transformers).

 Question: How many FLOPs does LLaMA-3 perform per token per 训练 step? This helps us determine how expensive the whole 训练 process will be.

 Click here for the answer, once you’ve thought about it! Answer: As shown in [Section 4](../transformers), we do roughly \(6 \cdot \text{param count}\) FLOPs per token, so here that’s roughly `6 * 70e9 = 4.2e11` FLOPs / token. That’s about half a TFLOP per token per step. Assuming we’re compute-bound, this should take roughly `4.2e11 / 4.59E+14 = 1ms` on a single TPU v5p 芯片, assuming perfect FLOPs utilization.

  Question: LLaMA 3 was trained for about 15 trillion tokens. How many FLOPs is that total?

 Click here for the answer, once you’ve thought about it! Answer: That’s easy, it’s just `4.2e11 * 15e12 = 6.3e24 FLOPs` total. 6.3 yottaFLOPs. That’s a lot! On a single TPU this would take `6.3e24 / 4.59E+14 = 435 years`. That’s also a lot!

  Question: Let’s say we wanted to train on a full TPU v5p pod with 16x20x28 = 8960 chips. How long would this take to train at 40% MFU in bfloat16, assuming we are compute-bound?

 Click here for the answer, once you’ve thought about it! Answer: We know that each TPU v5p can perform 4.59e14 FLOPs / second. At 40% MFU, this will take about `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 seconds`. This is about 44 days! That’s fairly reasonable, assuming we can actually achieve 40% MFU.

  Question: LLaMA 3-70B was pretrained with a batch size of about 4M tokens. How many TPUs do we need at minimum to train with this batch size? You can assume bfloat16 parameters and float32 优化器 state, and that you checkpoint gradients 4 times per 层.

 Click here for the answer, once you’ve thought about it! Answer: This question is primarily asking about 内存 usage, since that’s the only strict constraint on available compute. During 训练, we have three primary uses of HBM: 模型 parameters, 优化器 state, and 梯度 checkpoints. If we assume bfloat16 weights, float32 优化器 state, and a very conservative 梯度 checkpointing scheme (4 times per 层), we have:

    Params 2 * 70GB ~140GB   优化器 State 8 * 70GB ~560GB   梯度 Checkpoints 2 * 8192 * 4e6 * 4 * 80 ~20.9TB   Total   ~21.6TB    The total here is about 21.6TB. You notice that 梯度 checkpointing strongly dominates the 内存 picture, even with a very conservative checkpointing scheme. We could technically go to 1 checkpoint per 层, or do microbatching, but this is a reasonable picture. With these assumptions, since each TPU v5p has 96GB of HBM, we need `21.6e12 / 96e9 = 225` TPUs. That’s not very much actually!

 Why wouldn’t we do this? Well, because it would take us `44 days * 8960 / 225 = 1752 days` to train. That’s nearly four years. That’s a lot. Still, this makes it clear that we’re using these large clusters not because we’re bound by 内存 but rather because we need the extra FLOPs.

  Question: Under the same assumptions as the question above, if we use 8960 TPU v5p chips, how much 内存 will we use per-芯片?

 Click here for the answer, once you’ve thought about it! Answer: Our total 内存 is still about 21.6TB, so per-芯片 we’ll be using about 2.4GB per 芯片, which is basically nothing. If we did much more aggressive checkpointing, e.g. 12 checkpoints per 层, we’d still only be at 8GB per 芯片. We’re nowhere near being 内存 bound during 训练 at these scales.

  Takeaways: It is technically possible to train even very large models on very small topologies, with the caveat that they will likely take a long time. Being able to calculate the total FLOPs of a 训练 run allows us to ballpark its 训练 time by assuming a modest MFU and a known topology.

 ### How to shard LLaMA 3-70B for 训练 Let’s stick to our setting from above and say we want to train LLaMA 3-70B with 4M token batch size (1024 sequences of length 4096 per batch) on a TPU v5p pod of 8960 chips. Let’s discuss what the best 分片 strategy is for this 模型.

 Question: Under the assumptions above, can we train our 模型 with FSDP alone? To start, let’s say we can’t do any sequence/context parallelism. This should be the first idea you have, since it’s simple and will introduce no extra communication if it works.

 Click here for the answer, once you’ve thought about it! Answer: This answer will be a little pedantic. As noted above, LLaMA 3-70B is initially trained with sequences of length 4K, so the batch size of 4M tokens gives us a sequence batch size of 1024. That means we can only really do pure data parallelism/FSDP up to 1024 chips because that’s how many sequences we have to do data parallelism over. So the answer in the simple sense of “fully data parallelism with no extra communication” is no. The next question will answer a slightly less pedantic version of this.

  Question: Let’s relax the requirement of not doing any sequence 分片. If we allow ourselves to do FSDP over both the batch and sequence axes, can we train LLaMA 3-70B with only FSDP on 8960 chips?

 Click here for the answer, once you’ve thought about it! Answer: Now that we’re allowing ourselves to do sequence/context parallelism as well, we can scale up way more. First let’s calculate our per-device batch size. If we do 8960-way FSDP, we end with a per-TPU batch size of `4 * 1024 * 1024 / 8960 = 468 tokens`. We know from the previous section that we become ICI-bound by FSDP when \(\text{per device batch size} &lt; 2550 / M_X\). Since we can dedicate 3 axes here with a full 3D pod, this would give us a lower bound of 850, which we’re well below. So the answer is no, even with 3 axes. We would be solidly communication-bound.

  Question: Now let’s look at mixed 张量 parallelism and FSDP. Does there exist some combination that lets us remain compute-bound? What amount of FSDP and 张量 parallelism should we do if so?

 Click here for the answer, once you’ve thought about it! Answer: First let’s check to see if this will even fit. We know that we’ll be comms-bound if our per-芯片 batch size is less than $2550^2 / 2F = 113$. As we saw above, we’re slightly above this. So that’s great! Now to pick the optimal amount of FSDP, we can use the formula

 \[X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618\] Rounding to a reasonable multiple of 2, that gives us roughly 2048-way FSDP and 4-way 张量 parallelism. That should work well!

  Takeaways: We can train LLaMA-3 with a 4M token batch size on a full TPU v5p pod with a mixture of data parallelism (1024-way), sequence parallelism (2-way), and 张量 parallelism (4-way) without being communication-bound. We will be comms-bound if we try to do pure FSDP or FSDP + sequence parallelism. The equations we’ve cooked up in the previous section are very practical.

 ## Worked Problems Question 1 [Scaling LLaMA 70B to more chips]: say we want to train LLaMA 3-70B on 4 pods with the same batch size. What parallelism scheme would we use? Would we be compute or communication bound? Roughly how long would it take to train? Make sure to use the correct roofline bound.

 Question 2 [LLaMA 405B]:

 (a) Using the LLaMA 3-405B [config](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json), write a table with all the key hyperparameters as above. How many total parameters does this 模型 have? How many FLOPs per 训练 step? How many FLOPs do we perform if we train for 15T tokens?

 (b) Assume we want to train on 8 TPU v5p pods. What parallelism scheme would we use? How long would 训练 take? Would we be compute or comms bound?

 ### That’s all for Section 6. For Section 7, about Transformer 推理, click [here](../推理).      ### Miscellaneous *Work done at Google DeepMind, now at MatX.

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
