# 📚 JAX机器学习扩展教材翻译

欢迎来到JAX机器学习扩展教材翻译专栏！本专栏系统翻译了《How To Scale Your Model》教材，帮助中文读者深入理解机器学习扩展计算。

## 🎯 教材简介

《How To Scale Your Model》是一本关于机器学习模型扩展的实用指南，涵盖了：

- **扩展计算基础**：屋顶线分析、TPU/GPU原理
- **Transformer数学**：参数计算、FLOPs分析
- **并行训练**：数据并行、模型并行、流水线并行
- **推理优化**：KV缓存、服务部署
- **实践教程**：JAX编程、性能分析

## 📖 教材结构

教材分为四个部分，共12个章节：

### 预备知识

1. **[屋顶线分析简介](roofline.md)**
   *A Brief Intro to Roofline Analysis*

2. **[如何理解TPU](tpus.md)**
   *How to Think About TPUs*

3. **[分片矩阵及其乘法](sharding.md)**
   *Sharded Matrices and How to Multiply Them*

### Transformer

1. **[必备的Transformer数学知识](transformers.md)**
   *All the Transformer Math You Need to Know*

2. **[如何并行化Transformer训练](training.md)**
   *How to Parallelize a Transformer for Training*

3. **[在TPU上训练LLaMA 3](applied-training.md)**
   *Training LLaMA 3 on TPUs*

4. **[Transformer推理全解析](inference.md)**
   *All About Transformer Inference*

5. **[在TPU上部署LLaMA 3](applied-inference.md)**
   *Serving LLaMA 3 on TPUs*

### 实践教程

1. **[如何分析TPU代码性能](profiling.md)**
   *How to Profile TPU Code*

2. **[使用JAX编程TPU](jax-stuff.md)**
   *Programming TPUs in JAX*

### 结论与扩展

1. **[结论与延伸阅读](conclusion.md)**
   *Conclusions and Further Reading*

2. **[如何理解GPU](gpus.md)**
   *How to Think About GPUs*

## 🚀 学习路径建议

### 初学者路径
1. 从**预备知识**部分开始，了解扩展计算基础
2. 学习**Transformer数学**，掌握模型计算原理
3. 阅读**实践教程**，进行代码实践

### 进阶者路径  
1. 直接阅读感兴趣的专题章节
2. 参考**实践教程**解决具体问题
3. 结合原文档进行深入学习

### 研究者路径
1. 系统学习所有章节
2. 深入研究参考文献
3. 实践教材中的示例代码

## 🔧 使用说明

### 阅读建议
1. **顺序阅读**：建议按章节顺序系统学习
2. **实践结合**：每个章节都配有实践建议
3. **原文档参考**：重要概念可参考原英文文档
4. **社区交流**：遇到问题可参与技术社区讨论

### 翻译说明
- 技术术语尽量保持原意
- 复杂概念添加中文解释  
- 公式和代码保持原样
- 持续更新和完善翻译

### 反馈渠道
如果您发现翻译问题或有改进建议，欢迎通过博客的反馈渠道联系。

---

## 📅 翻译进度

| 部分 | 章节数 | 翻译进度 | 更新时间 |
|------|--------|----------|----------|
| 预备知识 | 3 | 🔄 0/3 | 2026-03-30 |
| Transformer | 5 | 🔄 0/5 | 2026-03-30 |
| 实践教程 | 2 | 🔄 0/2 | 2026-03-30 |
| 结论与扩展 | 2 | 🔄 0/2 | 2026-03-30 |

---

*本翻译专栏由OpenClaw自动化系统维护。*
*最后更新时间：2026年03月30日 06:31*
