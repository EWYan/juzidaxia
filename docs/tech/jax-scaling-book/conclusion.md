---
title: "结论与延伸阅读"
date: 2026-03-30
description: "结论与延伸阅读 - Conclusions and Further Reading"
categories:
  - JAX
  - 机器学习
  - 扩展计算
tags:
  - 结论与扩展
  - 翻译
---

# 结论与延伸阅读

*Conclusions and Further Reading*

## 📋 章节概览

**所属部分**：结论与扩展
**原文标题**：Conclusions and Further Reading
**原文地址**：https://jax-ml.github.io/scaling-book/conclusion
**翻译时间**：2026年03月30日

## 🎯 本章要点

本章将深入探讨结论与延伸阅读的相关内容，包括：

1. **核心概念**：理解结论与延伸阅读的基本原理
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
Conclusions and Further Reading | How To Scale Your 模型     *                         [ How To Scale Your 模型 ](/scaling-book)  Toggle navigation     [](../jax-stuff) [](../gpus)  
  [ ](/scaling-book/) 
 * [Previous Part](../jax-stuff)
 * [Next Part](../gpus)
 *  [  Sections ](#)  [Part 0. Introduction](/scaling-book/index) [Part 1. Intro to Rooflines](/scaling-book/roofline) [Part 2. All About TPUs](/scaling-book/tpus) [Part 3. Sharded Matmuls](/scaling-book/分片) [Part 4. Transformers](/scaling-book/transformers) [Part 5. 训练](/scaling-book/训练) [Part 6. 训练 LLaMA](/scaling-book/applied-训练) [Part 7. 推理](/scaling-book/推理) [Part 8. Serving LLaMA](/scaling-book/applied-推理) [Part 9. Profiling](/scaling-book/profiling) [Part 10. All About JAX](/scaling-book/jax-stuff) [Part 11. Conclusions](/scaling-book/conclusion) [Part 12. GPUs](/scaling-book/gpus)  
 *       
 
            # Conclusions and Further Reading Part 11 of [How To Scale Your 模型](/scaling-book) ([Part 10: JAX](../jax-stuff) | [Part 12: GPUs](../gpus))

 Thank you for reading! Here we'll include a few more references for further study.

      ### Contents  [Acknowledgments](#acknowledgments)   [Further Reading](#further-reading)   [Feedback](#feedback)    Thank you for reading the whole thing and congratulations on making it all the way to the end. Before we conclude, a few acknowledgments:

 ## Acknowledgments This document represents a significant collective investment from many people at Google DeepMind, who we’d like to briefly acknowledge!

 * James Bradbury, Reiner Pope, and Blake Hechtman originally derived many of the ideas in this manuscript, and were early to understanding the systems view of the Transformer.
 * Sholto Douglas wrote the first version of this doc and is responsible for kicking off the project. He is more than anyone responsible for the overall narrative of this doc.
 * Jacob Austin led the work of transforming this first version from rough notes into a more polished and comprehensive artifact. He did much of the work of editing, formatting, and releasing this document, and coordinated contributions from other authors.
 * Most of the figures and animations were made by Anselm Levskaya and Charlie Chen.
 * Charlie Chen wrote the 推理 section and drew many of the 推理 figures.
 * Roy Frostig helped with publication, editing, and many other steps of the journey.
 
 We’d also like to thank many others who gave critical feedback throughout the process, in particular Zak Stone, Nikhil Sethi, Caitlin Stanton, Alek Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins, and many others. Thanks to Ruiqi Gao for help with the HTML formatting.

 Thank you all!

 Before you go, you might also enjoy reading the new [Part 12](../gpus) on NVIDIA GPUs!

 ## Further Reading There is a bunch of related writing, including the following:

 *  [TPU Deep Dive](https://henryhmko.github.io/posts/tpu/tpu.html): a wonderful in-depth look at the TPU 架构 in the spirit of this book.
 *  [Domain specific architectures for AI 推理](https://fleetwood.dev/posts/domain-specific-architectures): a hardware and 模型 deep dive in the spirit of this book.
 *  [A Domain-Specific Supercomputer for 训练 Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3360307): one of the OG TPU papers, this has a lot of great details about the Google TPU program not covered here.
 *  [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): a more GPU and PyTorch-focused tutorial on LLM rooflines and 性能 engineering.
 *  [Writing TPU Kernels with Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): increasingly, TPU programming involves writing custom kernels in Pallas. This series discusses how to write kernels and many lower level TPU details that aren’t mentioned here.
 *  [How to Optimize a CUDA Matmul Kernel for cuBLAS-like 性能: a Worklog](https://siboehm.com/articles/22/CUDA-MMM): while GPU and CUDA specific, this is an excellent blog post showing how to optimize a matmul kernel in CUDA. This might be a good deep dive into how TPUs and GPUs are different.
 *  [分布式 arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): this is a really nice guide to parallelism APIs in JAX and is a good way to learn how to actually implement some of the ideas we’ve discussed here.
 *  [Rafi Witten’s High 性能 LLMs 2024 Class](https://github.com/rwitten/HighPerfLLMs2024): our former colleague Rafi gave a great course on TPU 性能 engineering and the slides are all on GitHub. This covers a bunch of things in more depth than we do here.
 *  [[2211.05102] Efficiently Scaling Transformer 推理](https://arxiv.org/abs/2211.05102): a detailed paper on the mathematics of Transformer 推理. This is the inspiration for a lot of this document.
 *  [Huggingface Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook): something of a GPU analog to this book, this talks more at depth about how PyTorch implements parallelism techniques and 内存-saving techniques during 训练.
 *  [Transformer 推理 Arithmetic](https://kipp.ly/transformer-推理-arithmetic/): a blog with many of the same ideas as this book and some excellent illustrations.
 *  [Stanford CS336 Slides and Videos](https://stanford-cs336.github.io/spring2025/index.html#coursework): a fantastic Stanford course covering many details of LLM 训练 and serving, with some useful exercises. Assignments 1 and 2 are particularly relevant.
 *  [Stas Bekman’s ML Engineering Handbook](https://github.com/stas00/ml-engineering): a highly practical guide to ML infrastructure, covering topics not addressed in this book like how to negotiate with cloud providers, cluster management, and empirical measurements of GPU 吞吐量.
 
 There remains a lot of room for comprehensive writing in this area, so we hope this manuscript encourages more of it! We also believe that this is a fruitful area to study and research. In many cases, it can be done even without having many hardware accelerators on hand.

 ## Feedback Please leave comments or questions so that we can improve this further. You can reach our corresponding author, Jacob Austin, at jacobaustin123 [at] gmail [dot] com, or suggest edits by posting issues, pull requests, or discussions [on GitHub](https://github.com/jax-ml/scaling-book).

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
