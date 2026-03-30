---
title: "如何理解TPU - How to Think About TPUs"
date: 2026-03-30
description: "如何理解TPU - JAX机器学习扩展教材第2章"
categories:
  - JAX
  - 机器学习
  - 扩展计算
  - 翻译
tags:
  - JAX
  - TPU
  - GPU
  - Transformer
  - 并行计算
  - 机器学习扩展
---

# 如何理解TPU
*How to Think About TPUs*

## 📖 章节信息

- **所属部分**：预备知识
- **原文章节**：How to Think About TPUs
- **翻译时间**：2026年03月30日
- **原文地址**：https://jax-ml.github.io/scaling-book/tpus

## 🎯 学习目标

通过本章学习，您将了解：

1. **核心概念**：掌握如何理解TPU的基本原理
2. **实践应用**：了解在实际项目中的应用方法
3. **技术细节**：深入理解相关的技术实现
4. **最佳实践**：学习行业内的最佳实践方法

---

---
**翻译说明**：
- 本文为《How to Think About TPUs》的中文翻译
- 原文档地址：https://jax-ml.github.io/scaling-book/how-to-think-about-tpus
- 翻译时间：2026年03月30日
- 翻译状态：初步翻译，正在完善中

**注意事项**：
1. 技术术语尽量保持原意
2. 复杂概念添加中文解释
3. 公式和代码保持原样
4. 图表引用原文档

---

How to Think About TPUs | How To Scale Your Model     *                         [ How To Scale Your Model ](/scaling-book)  Toggle navigation     [](../roofline) [](../sharding)    [ ](/scaling-book/) 
 * [Previous Part](../roofline)
 * [Next Part](../sharding)
 *  [  Sections ](#)  [Part 0. Introduction](/scaling-book/index) [Part 1. Intro to Rooflines](/scaling-book/roofline) [Part 2. All About TPUs](/scaling-book/tpus) [Part 3. Sharded Matmuls](/scaling-book/sharding) [Part 4. Transformers](/scaling-book/transformers) [Part 5. Training](/scaling-book/training) [Part 6. Training LLaMA](/scaling-book/applied-training) [Part 7. Inference](/scaling-book/inference) [Part 8. Serving LLaMA](/scaling-book/applied-inference) [Part 9. Profiling](/scaling-book/profiling) [Part 10. All About JAX](/scaling-book/jax-stuff) [Part 11. Conclusions](/scaling-book/conclusion) [Part 12. GPUs](/scaling-book/gpus)  
 *       
 
            # How to Think About TPUs Part 2 of [How To Scale Your Model](/scaling-book) ([Part 1: Rooflines](../roofline) | [Part 3: Sharding](../sharding))

 This section is all about how TPUs work, how they're networked together to enable multi-chip 训练 and 推理, and how this affects the performance of our favorite algorithms. There's even some good stuff for GPU users too!

      ### Contents  [What Is a TPU?](#what-is-a-tpu)   [TPU Networking](#tpu-networking)   [Key Takeaways](#key-takeaways)   [](#)   *  [TPU specs](#tpu-specs) 
 
  [Worked Problems](#worked-problems)   [Appendix](#appendix)   [](#)   *  [Appendix A: More on TPU internals](#appendix-a-more-on-tpu-internals) 
 *  [Appendix B: How does a systolic array work?](#appendix-b-how-does-a-systolic-array-work) 
 
   You might also enjoy reading the new [Section 12](../gpus) on NVIDIA GPUs!

 ## What Is a TPU? A TPU is basically a compute core that specializes in 矩阵 乘法 (called a TensorCore) attached to a stack of fast 内存 (called high-bandwidth 内存 or HBM). Here’s a diagram:

      Figure: the basic components of a TPU chip. The TensorCore is the gray left-hand box, containing the matrix-multiply unit (MXU), vector unit (VPU), and vector 内存 (VMEM).  You can think of the TensorCore as basically just being a really good 矩阵 乘法 machine, but it has a few other functions worth noting. The TensorCore has three key units:

  * The MXU (Matrix Multiply Unit) is the core of the TensorCore. For most TPU generations, it performs one `bfloat16[8,128] @ bf16[128,128] -&gt; f32[8,128]` 矩阵 multiplyTPU v6e (Trillium) has a 256x256 MXU, while all previous generations use 128x128 every 8 cycles using a systolic array (see [Appendix B](#appendix-b-how-does-a-systolic-array-work) for details).  This is about `5e13` bf16 FLOPs/s per MXU at 1.5GHz on TPU v5e. Most TensorCores have 2 or 4 MXUs, so e.g. the total bf16 FLOPs/s for TPU v5e is `2e14`.
 * TPUs also support lower precision matmuls with higher throughput (e.g. each TPU v5e chip can do `4e14` int8 OPs/s).
 
  * The VPU (Vector Processing Unit) performs general mathematical operations like ReLU activations or pointwise addition or 乘法 between vectors. Reductions (sums) are also performed here. [Appendix A](#appendix-a-more-on-tpu-internals) provides more details.
 *  VMEM (Vector Memory) is an on-chip scratchpad located in the TensorCore, close to the compute units. It is much smaller than HBM (for example, 128 MiB on TPU v5e) but has a much higher 带宽 to the MXU. VMEM operates somewhat like an L1/L2 cache on CPUs but is much larger and programmer-controlled. Data in HBM needs to be copied into VMEM before the TensorCore can do any 计算 with it.
 
 TPUs are very, very fast at 矩阵 乘法. It’s mainly what they do and they do it well. [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture), one of the most powerful TPUs to date, can do `2.5e14` bf16 FLOPs / second / core or `5e14` bf16 FLOPs / sec / chip. A single pod of 8960 chips can do 4 exaflops / second. That’s a lot. That’s one of the most powerful supercomputers in the world. And Google has a lot of them.TPUs, and their systolic arrays in particular, are such powerful hardware accelerators because 矩阵 乘法 is one of the few algorithms that uses $O(n^3)$ compute for $O(n^2)$ bytes. That makes it very easy for an ordinary ALU to be bottlenecked by compute and not by 内存 带宽.

 The diagram above also includes a few other components like SMEM and the scalar unit, which are used for control flow handling and are discussed briefly in [Appendix A](#appendix-a-more-on-tpu-internals), but aren’t crucial to understand. On the other hand, HBM is important and fairly simple:

   HBM (High Bandwidth Memory) is a big chunk of fast 内存 that stores tensors for use by the TensorCore. HBM usually has capacity on the order of tens of gigabytes (for example, [TPU v5e has 16GiB of HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)).

   When needed for a 计算, tensors are streamed out of HBM through VMEM (see below) into the MXU and the result is written from VMEM back to HBM.

   The 带宽 between HBM and the TensorCore (through VMEM) is known as “HBM bandwidth” (usually around 1-2TB/sec) and limits how fast 计算 can be done in memory-bound workloads.

 Generally, all TPU operations are pipelined and overlapped. To perform a matmul $X \cdot A \to Y$, a TPU would first need to copy chunks of matrices $A$ and $X$ from HBM into VMEM, then load them into the MXU which multiplies chunks of 8x128 (for $X$) and 128x128 (for $A$), then copy the result chunk by chunk back to HBM. To do this efficiently, the matmul is pipelined so the copies to/from VMEM are overlapped with the MXU work. This allows the MXU to continue working instead of waiting on 内存 transfers, keeping matmuls compute-bound, not memory-bound.

 Here’s an example of how you might perform an elementwise product from HBM:

      Figure: an animation showing a pointwise product performed on TPU, with bytes loaded from HBM. Note how bytes are streamed out of 内存 in chunks and partial results are pipelined back without waiting for the full array to be materialized.  A matmul would look nearly identical except it would load into the MXU instead of the VPU/Vector unit, and the loads and stores would occur in a different order, since the same weight chunk is used for multiple chunks of activations. You can see chunks of data streaming into VMEM, then into the VREGs (vector registers), then into the Vector Unit, then back into VMEM and HBM. As we’re about to see, if the load from HBM to VMEM is slower than the FLOPs in the Vector Unit (or MXU), we become “bandwidth bound” since we’re starving the VPU or MXU of work.

 Key takeaway: TPUs are very simple. They load weights from HBM into VMEM, then from VMEM into a systolic array which can perform around 200 trillion multiply-adds per second. The HBM $\leftrightarrow$ VMEM and VMEM $\leftrightarrow$ systolic array bandwidths set fundamental limits on what computations TPUs can do efficiently.

 VMEM and arithmetic intensity: VMEM is much smaller than HBM but it has a much higher 带宽 to the MXU. As we saw in [Section 1](../roofline), this means if an algorithm can fit all its inputs/outputs in VMEM, it’s much less likely to hit 通信 bottlenecks. This is particularly helpful when a 计算 has poor arithmetic intensity: VMEM 带宽 is around 22x higher than HBM 带宽 which means an MXU operation reading from/writing to VMEM requires an arithmetic intensity of only 10-20 to achieve peak FLOPs utilization. That means if we can fit our weights into VMEM instead of HBM, our 矩阵 multiplications can be FLOPs bound at much smaller batch sizes. And it means algorithms that fundamentally have a lower arithmetic intensity can still be efficient. VMEM is just so small this is often a challenge.We sometimes talk about VMEM prefetching, which refers to loading weights ahead of time in VMEM so we can mask the cost of loading for our matmuls. For instance, in a normal Transformer we can sometimes load our big feed-forward weights into VMEM during attention, which can hide the cost of the weight load if we're 内存 带宽 bound. This requires our weights to be small enough or sharded enough to fit a single layer into VMEM with space to spare.

       A TPU chip typically (but not always) consists of two TPU cores which share 内存 and can be thought of as one large accelerator with twice the FLOPs (known as a “megacore” configuration). This has been true since TPU v4. Older TPU chips have separate 内存 and are regarded as two separate accelerators (TPU v3 and older). Inference-optimized chips like the TPU v5e only have one TPU core per chip.

       Chips are arranged in sets of 4 on a ‘tray’ connected to a CPU host via PCIe network. This is the format most readers will be familiar with, 4 chips (8 cores, though usually treated as 4 logical megacores) exposed through Colab or a single TPU-VM. For 推理 chips like the TPU v5e, we have 2 trays per host, instead of 1, but also only 1 core per chip, giving us 8 chips = 8 cores.On Cloud TPU VMs, each tray is exposed as part of a separate VM, so there are once again 4 cores visible.

       PCIe 带宽 is limited: Like the HBM $\leftrightarrow$ VMEM link, the CPU $\leftrightarrow$ HBM PCIe connection has a specific 带宽 that limits how quickly you can load from host 内存 to HBM or vice-versa. PCIe 带宽 for TPU v4 is 16GB / second each way, for example, so close to 100x slower than HBM. We can load/offload data into the host (CPU) RAM, but not very quickly.

 ## TPU Networking Chips are connected to each other through the ICI network in a Pod. In older generations (TPU v2 and TPU v3), 推理 chips (e.g., TPU v5e), and Trillium (TPU v6e), ICI (“inter-chip interconnects”) connects the 4 nearest neighbors (with edge links to form a 2D torus). TPU v4 and TPU v5p are connected to the nearest 6 neighbors (forming a 3D torus). Note these connections do not go through their hosts, they are direct links between chips.

       The toroidal structure reduces the maximum distance between any two nodes from $N$ to $N / 2$, making 通信 much faster. TPUs also have a “twisted torus” configuration that wraps the torus in a Mobius-strip like topology to further reduce the average distance between nodes.

 TPU pods (connected by ICI) can get really big: the maximum pod size (called a superpod) is `16x16x16` for TPU v4 and `16x20x28` for TPU v5p. These large pods are composed of reconfigurable cubes of `4x4x4` chips connected by [optical wraparound links](https://arxiv.org/pdf/2208.10041)The optical switch is simply a reconfigurable connection with the same ICI 带宽. It just lets us connect cubes while retaining a wraparound link. that we can reconfigure to connect very large topologies.

       Smaller topologies (e.g. `2x2x1`, `2x2x2`) can also be requested, albeit with no wraparounds. This is an important caveat, since it typically doubles the time of most 通信. Any multiple of a full cube (e.g. `4x4x4` or `4x4x8`) will have wraparounds provided by the optical switches.Note that a `2x2x4` won't have any wraparounds since they are provided by the optical switches which are only available on a full cube. A TPU v5e 8x16 _will_ have a wraparound on the longer axis, however, since it doesn't use reconfigurable optical networking.

       TPU v5e and Trillium pods consist of a single `16x16` 2D torus with wraparounds along any axis of size 16 (meaning an `8x16` has a wraparound on the long axis). TPUs v5e and v6e (Trillium) cannot expand beyond a 16x16 torus but pods can still communicate with each other over standard data-center networking (DCN), which connects TPU hosts to each other. Again, smaller topologies can be requested without wraps on dims $&lt;16$.

       This nearest-neighbor connectivity is a key difference between TPUs and GPUs. GPUs are connected with a hierarchy of switches that approximate a point-to-point connection between every GPU, rather than using local connections like a TPU. Typically, GPUs within a node (8 GPUs for H100 or as many as 72 for B200 NVL72) are directly connected, while larger topologies require O(log(N)) hops between each GPU. On the one hand, that means GPUs can send arbitrary data within a small number of hops. On the other hand, TPUs are dramatically cheaper (since NVLink switches are expensive), simpler to wire together, and can scale to much larger topologies because the number of links per device and the 带宽 per device is constant. Read more [here](../gpus#networking).

 ICI is very fast relative to DCN, but is still slower than HBM 带宽. For instance, a [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) has:

  *  `2.8e12` bytes/s (2.8 TB/s) of HBM 带宽 per chip.
 *  `9e10` bytes/s (90 GB/s) of ICI 带宽 per axis, with 3 axes per chip.The page above lists 100 GB/s of 带宽, which is slightly different from what's listed here. TPU ICI links have slightly different bandwidths depending on the operation being performed. You can generally use the numbers in this doc without worry. 
 *  `6.25e9` bytes/s (6.25 GB/s) of DCN (egress) 带宽 per TPU (via 1-2 NICs on each host).TPU v6e has 12.5e9 bytes/s and v5e has 3.125e9 bytes/s. 
 
 This means that when we split models across multiple chips, we need to be careful to avoid bottlenecking the MXU with slower cross-device 通信.

 Multi-slice training: A set of ICI-connected TPUs is called a slice. Different slices can be connected between each other using DCN, for instance to link slices on different pods. Since DCN is a much slower connection than ICI, we should try to limit how much our 计算 has to wait for data from DCN. DCN is host-to-host, so to transfer buffers from TPU to TPU over DCN, we first need to transfer over PCIe to the host, then egress over the network, then ingress over the target host network, then over PCIe into HBM.

 ## Key Takeaways   TPUs are simple and can in most cases be thought of as a 矩阵 multiply unit connected to 内存 (super fast), other chips over ICI (rather fast), and the rest of the datacenter over DCN (somewhat fast).

  * Communication is limited by our various network bandwidths in order of speed:  HBM bandwidth: Between a TensorCore and its associated HBM.
 * ICI bandwidth: Between a TPU chip and its nearest 4 or 6 neighbors.
 * PCIe bandwidth: Between a CPU host and its associated tray(s) of chips.
 * DCN bandwidth: Between multiple CPU hosts, typically hosts not connected by ICI.
 
   Within a slice, TPUs are only connected to their nearest neighbors via ICI. This means 通信 over ICI between distant chips in a slice needs to hop over the intervening chips first.

   Weight matrices need to be padded to at least size 128 (256 on TPU v6e) in both dimensions to fill up the MXU (in fact, smaller axes are padded to 128).

   Lower precision 矩阵 乘法 tends to be faster. TPUs can do int8 or int4 FLOPs roughly 2x/4x faster than bfloat16 FLOPs for generations that support it. VPU operations are still performed in fp32.

  * To avoid bottlenecking the TPU compute unit, we need to make sure the amount of 通信 across each channel is proportional to its speed.
 
 ### TPU specs Here are some specific numbers for our chips:

    Model Pod size Host size HBM capacity/chip HBM BW/chip (bytes/s) FLOPs/s/chip (bf16) FLOPs/s/chip (int8)     TPU v3 32x32 4x2 32GB 9.0e11 1.4e14 1.4e14   TPU v4p 16x16x16 2x2x1 32GB 1.2e12 2.75e14 2.75e14   TPU v5p 16x20x28 2x2x1 96GB 2.8e12 4.59e14 9.18e14   TPU v5e 16x16 4x2 16GB 8.1e11 1.97e14 3.94e14   TPU v6e 16x16 4x2 32GB 1.6e12 9.20e14 1.84e15    Host size refers to the topology of TPUs connected to a single host (e.g. TPU v5e has a single CPU host connected to 8 TPUs in a 4x2 topology). Here are interconnect figures:

    Model ICI BW/link (one-way, bytes/s) ICI BW/link (bidi, bytes/s)     TPU v3 1.0e11 2.0e11   TPU v4p 4.5e10 9.0e10   TPU v5p 9.0e10 1.8e11   TPU v5e 4.5e10 9.0e10   TPU v6e 9.0e10 1.8e11    We include both one-way (unidirectional) 带宽 and bidi (bidirectional) 带宽 since unidirectional 带宽 is more true to the hardware but bidirectional 带宽 occurs more often in equations involving a full ring.By bidi (bidirectional) 带宽 we mean the total bytes that can be sent along a single link in both directions, or equally, the total number of outgoing bytes from a single TPU along a particular axis, assuming we can use both links efficiently. This is true when we have a functioning ring, AKA when we have a wraparound connection on the particular axis. This occurs on 推理 chips when we have a full 16 axis, or on 训练 chips (v*p) when we have an axis which is a multiple of 4. We prefer to use the bidirectional 带宽 because it appears frequently in calculations involving bidirectional comms.

 PCIe 带宽 is typically around `1.6e10` bytes / second per TPU (`3.2e10` for TPU v6e), while DCN 带宽 is typically around `6.25e9` bytes / second per TPU (`12.5e9` for TPU v6e and `3.125e9` for TPU v5e).

 ## Worked Problems These numbers are a little dry, but they let you make basic roofline estimates for model performance. Let’s work a few problems to explain why this is useful. You’ll see more examples in Part 3.

 Question 1 [bounding LLM latency]: Say you want to sample from a 200B parameter model in bf16 that’s split across 32 TPU v4p. How long would it take to load all the parameters from HBM into the systolic array? Hint: use the numbers above.

 Click here for the answer. Answer: We’re loading `sizeof(bf16) * 200e9 = 400e9` bytes on 32 chips, meaning 12.5e9 bytes / chip, each with an HBM 带宽 of 1.23e12. So the load takes around 10ms.

 That’s pretty cool, because that’s a reasonable lower bound on the latency of sampling from the model. Each sampling step needs to load all parameters from HBM, so it cannot take less than 10 ms. In practice, at small batch sizes, this is close to being achievable.

  Question 2 [TPU details]: Consider a full TPU v5e pod. How many total CPU hosts are there? How many TPU TensorCores? What is the total FLOPs/s for the whole pod? What is the total HBM? Do the same exercise for TPU v5p pod.

 Click here for the answer. Answer: For TPU v5e, each pod is `16x16` and each host is a 4x2 slice, so we have `16*16 / 8 = 32` hosts. For TPU v5e, each TPU has only one core, so we have 256 TensorCores. The total FLOPs/s is `16*16*2e14 = 5.1e16` in bfloat16. Each chip has 16GB of HBM, so that’s `256 * 16 = 4TB` of 内存.

 For a full TPU v5p pod, we have `16x20x28` chips and each host is 2x2x1, so we have `(16*20*28) / (2*2) = 2,240` hosts. For TPU v5p, each TPU has two TensorCores, so we have `8960 * 2 = 17,920` cores. The total FLOPs/s is `8960 * 4.5e14 = 4e18` in bfloat16. Each chip has 96GB of HBM, so that’s `8960 * 96 = 860TB` of 内存.

  Question 3 [PCIe operational intensity]: Imagine we’re forced to store a big weight 矩阵 $A$ of type $\text{bfloat16}[D, F]$, and a batch of activations $x$ of type $\text{bfloat16}[B, D]$ in host DRAM and want to do a 矩阵 乘法 on them. This is running on a single host, and we’re using a single TPU v6e chip attached to it. You can assume $B \ll D$, and $F = 4D$ (we’ll see in future chapters why these are reasonable assumptions). What is the smallest batch size $B$ we need to remain FLOPs bound over PCIe? Assume PCIe 带宽 of 1.5e10 bytes / second.

 Click here for the answer. Answer: We have to perform $2BDF$ floating point operations, and each chip can perform `9.2e14` floating point operations per second. This then requires $2BDF / 9.2e14$ seconds to perform. We have to load $2DF + 2BD$ bytes from DRAM, and write $2BF$ bytes back to it. We are bottlenecked by PCIe transfer speeds, so we need $2 \cdot (BD + DF + BF) / 1.5e10$ seconds to transfer data to and from the TPU. Since we want 计算 to take longer than weight loading, assuming we can overlap all weight loading with 计算, we want $2BDF / 9.2e14 &gt; 2 \cdot (BD + DF + BF) / 1.5e10$. We can simplify this using our assumptions that $B \ll D$, and $F = 4D$, to get

 \[\frac{8BD^2}{9.2 \times 10^{14}} &gt; \frac{8D^2}{1.5 \times 10^{10}}\] or

 \[B &gt; \frac{9.2 \times 10^{14}}{1.5 \times 10^{10}} \simeq 61{,}000\]  Question 4 [general matmul latency]: Let’s say we want to multiply a weight 矩阵 int8[16384, 4096] by an activation 矩阵 of size int8[B, 4096] where B is some unknown batch size. Let’s say we’re on 1 TPU v5e to start.

  * How long will this 乘法 take as a function of B? Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the 乘法 will actually take. Which is bottlenecking you? 
 * What if we wanted to run this operation out of VMEM? How long would it take as a function of B?
 
 Click here for the answer. Answer: (1) The number of floating point operations we need to perform is $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$. So $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ seconds. We need to load $16384 \cdot 4096 + 4096 \cdot B$ bytes from HBM to VMEM, and write back $16384 \cdot B$ bytes from VMEM to HBM. This means $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.1 \times 10^{11}$ seconds. Assuming as much overlap of 通信 and 计算 as possible, the whole 乘法 will take approximately

 \[\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{ \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}}, \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}} \right\}\] We’ll be FLOPs-bound when $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}} &lt; \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$, or equivalently, $B &gt; 271$. This is slightly larger than the 240 number we derive in [Section 1](../roofline) because we factor in the full impact of $D$ and $F$.

 (2) If instead we are loading from VMEM, let’s consider VMEM 带宽 to the MXU as 22 times the HBM $\leftrightarrow$ VMEM 带宽. This turns our data loading denominator from 8.1e11 to 1.78e13, and we get $B &gt; 11$. Note that in practice, we cannot dedicate all of our VMEM 带宽 to loading $W$, so in practice it will be closer to 20.

  Question 5 [ICI bandwidth]: Let’s say we have a TPU v5e `4x4` slice. Let’s say we want to send an array of type `bfloat16[8, 128, 8192]` from `TPU{0,0}` to `TPU{3, 3}`. Let’s say the per-hop latency for TPU v5e is $1\mu s$.

  * How soon will the first byte arrive at its destination?
 * How long will the total transfer take?
 
 Click here for the answer. Answer: In a TPU v5e we have 2D connectivity. Because we have only a `4x4` slice (with no axes of size 16), we have no wraparound connections. Thus there are two ports from which our target chip can receive data, and likewise two ports from which our source chip can send data. The amount of data we have to transfer is `2 * 8 * 128 * 8192 = 1.7e7` bytes. We can transfer from both ports simultaneously (i.e. send half the array right and half down), so we get `2 * 4.5e10 = 9e10` bytes transferred per second, which means it’ll take about `1.7e7 / 9e10 = 188us` to transfer the whole array through (assuming we’re 带宽 bound). In a `4x4` slice, we have six hops between chips $(0, 0)$ and $(3, 3)$, since there are no wraparound links for axes with fewer than 16 chips. Since the latency of each hop is about $1\mu s$, the first byte will arrive in about`6us` and the total transfer will take `188us`.

  Question 6 [pulling it all together, hard]: Imagine you have a big 矩阵 A: `int8[128 * 1024, 128 * 1024]` sharded evenly across a TPU v5e 4x4 slice but offloaded to host DRAM on each chip. Let’s say you want to copy the entire array to TPU{0, 0} and multiply it by a vector `bf16[8, 128 * 1024]`. How long will this take? Hint: use the numbers above.

 Click here for the answer. Answer: Let’s start by outlining the operations we have to perform. Our array is about 16GB. From the table above, a TPU v5e host has a 4x2 topology, so a 4x4 has 2 hosts. Thus, since our array is evenly sharded, each host effectively contains a chunk of 1/2 of the array, or 8GB. We need to copy these chunks all to TPU{0,0}, which gives us two options:

  * We can copy over DCN and then load the entire unsharded array over PCIe into HBM.
 * We can load our sharded arrays onto their corresponding TPUs, then perform a gather over ICI, then perform the matmul on TPU{0,0}.
 
 It should be clear that option (2) is better. DCN is slow compared to ICI and we’d much prefer to load a big array over many PCIe links rather than just a few (the 8 on host 0). Here’s a diagram of part of the system. As described above, note that TPUs are connected to their neighbors by ICI (even across hosts), all TPUs are connected to their host CPU (via PCIe), and hosts are connected by DCN.

      Each chip actually has its own PCIe link to its host, though for clarity only one is shown here.  Now let’s work through how long each piece will take:

   PCIe load: we’re loading chunks of 16GB over 16 PCIe links, each of which has `1.5e10` bytes/second 带宽. Thus this will take about 66ms.

   ICI copy: each TPU now has 16GB / 16 = 1GB of our array. Our ICI 带宽 is 9e10 bytes/second per link bidirectional, and you’ll notice from the above diagram that only 2 of the 4 ICI links on the TPU v5e are in use in this topology for TPU{0,0}. Since TPU{0,0} needs to receive a total of 15GB along 2 axes at `4.5e10` bytes/s/link, we can lower bound the time by `15e9 / (4.5e10 * 2) = 167ms`. In practice this probably isn’t achievable because the load is very uneven, but it’s probably within a factor of 2. As you’ll see in Section 3, performing a full AllGather would also take roughly `16e9 / (4.5e10 * 2)`, so this is close to optimal.

   HBM $\rightarrow$ MXU load: to perform our final matmul, we need to load these 16e9 bytes plus the bf16[8, 128 * 1024] array (another 2MB, so negligible) over HBM 带宽 into the MXU, which will take `16e9 / 8.1e11 = 19ms`.

   FLOPs: we’re performing a total of \(2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}\) FLOPs, and since we can perform `1.97e14` bf16 FLOPs/s, we get 1.3ms.

 An upper bound for the total time is the sum of all of these times, but since the TPU can typically overlap these operations, we can think of this as a pipelining problem that’s bottlenecked by the slowest piece. Assuming that’s true, then the answer is at least 167ms, likely closer to 200ms with imperfect overlapping.

  ### That’s it for Part 2! For Part 3, covering partitioning and cross-TPU 通信, [click here](../sharding). ## Appendix ### Appendix A: More on TPU internals Here we’ll dive more deeply into the internal operations of a TPU. Unless otherwise noted, we’ll provide specs for a TPU v5p.

 ### VPU The VPU is the TPU’s vector arithmetic core. The VPU consists of a two dimensional SIMD vector machine (the VPU) that performs elementwise arithmetic operations like vadd (vector addition) or vmax (elementwise max) and a set of vector registers called VREGs that hold data for the VPU and MXU.

 VREGs: Each TPU v5p core has 64 32-bit VREGs (32 in TPU v4), giving us a total of about `64 * 8 * 128 * 4 = 256kB` of VREG 内存 per core (or 2x this for the whole chip since we have two cores). A TPU v5p can load 3 registers from VMEM each cycle, and write 1 register to VMEM each cycle.

 VPU: The VPU is a 2D vector arithmetic unit of shape `(8, 128)` where the 128 dimension is referred to as lane axis and the dimension of 8 is referred to as the sublane axis. Each (lane, sublane) pair on v5 contains 4 standard floating-point ALUs which are independent of each other. The VPU executes most arithmetic instructions in one cycle in each of its ALUs (like vadd or vector add) with a latency of 2 cycles, so e.g. in v5 you can add 4 pairs of f32 values together from VREGs in each cycle. A typical VPU instruction might look like `{v2 = vadd.8x128.f32 v0, v1}` where v0 and v1 are input VREGs and v2 is an output VREG.

 All lanes and sublanes execute the same program every cycle in a pure SIMD manner, but each ALU can perform a different operation. So we can e.g. process 1 vadd and 1 vsub in a single cycle, each of which operates on two full VREGs and writes the output to a third.

 Pop Quiz [Calculating VPU throughput]: Using the above information, calculate how many vector FLOPs/s a TPU v5p can perform. A TPU v5p has a clock speed of about 1.75GHz.

 Click here for the answer. Answer: Each cycle, each core can execute 4 vector instructions on `8 * 128` ALUs. This gives us `8 * 128 * 4 * 2` FLOPs/cycle for the whole chip, or `8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`. Note how much smaller this is than the MXU FLOPs/s of about `2e14` (roughly 10x).

  Reductions: Generally, 通信 or reduction across the sublane dimension is easier than across the lane dimension. For instance, the VPU supports an intra-lane shuffle operation that can roll along the axis of size 8 in about a cycle. This can be used to perform efficient reductions along the sublane dimension (just shuffle by 4, 2, and 1 and do 3 pairs of elementwise sums).

 Cross-lane reductions are much harder and involve a separate hardware unit called the XLU or “cross lane unit”, which is slow and fairly expensive.

 Comparison to GPUs: For those familiar with NVIDIA GPUs, each ALU in the VPU is analogous to a CUDA core, and a single VPU lane is analogous to a “Warp Scheduler”, i.e. the set of usually 32 CUDA Cores that perform SIMD arithmetic. Reductions within the lane are pretty easy, but if we need to cross lanes, we need to transit at least VMEM/XLU/SMEM which is much slower. See the [GPU section](../gpus) for more details.

 ### Scalar Core The scalar core is the control unit of the TPU. It fetches and dispatches all instructions and executes transfers from HBM into VMEM, and can be programmed to do scalar metadata work. Because the scalar core is single-threaded, one side-effect of this is that each core of the TPU is only capable of creating one DMA request per cycle.

 To put this in context, a single scalar core controls a VPU (consisting of 4096 ALUs), 4 MXUs, 2 XLUs, and multiple DMA engines. The highly skewed nature of control per unit compute is a source of hardware efficiency, but also limits the ability to do data dependent vectorization in any interesting way.

 ### Appendix B: How does a systolic array work? At the core of the TPU MXU is a `128x128` systolic array (`256x256` on TPU v6e). When fully saturated the systolic array can perform one `bfloat16[8,128] @ bf16[128,128] -&gt; f32[8,128]`If you are not familiar with this notation, it means: multiplying a `8x128` 矩阵 with bfloat16 elements by a `128x128` 矩阵 with bfloat16 elements and storing the results in a `8x128` 矩阵 with float32 elements. 乘法 per 8 clock cycles.

  * At its core, the systolic array is a 2D `128x128` (`=16,384`) grid of ALUs each capable of performing a multiply and add operation.
 * Weights (W, the `128x128` input) are passed down from above (called the RHS) while inputs (X, the `8x128` input) are passed in from the left (called the LHS).
 
 Here is a simplified animation of multiplying a set of weights (blue) with a set of activations (green). You’ll notice that the weights (RHS) are partially loaded first, diagonally, and then the activations are fed in, also diagonally. In each frame below, we multiply all the overlapped green and blue units, sum the result with any residual passed in from above, and then pass the result in turn down one unit.

       Here’s a more general version of this animation showing the output being streamed out of computation:

       Here’s a diagram showing how this can be pipelined across multiple RHS and LHS arrays:

       There is an initial pipeline bubble as the weights (RHS) and activations (LHS) are loaded. After that initial bubble, new inputs and weights can be loaded in without an additional bubble.

 Here’s a bad animation of a bf16[2, 3] x bf16[3, 3] 矩阵 乘法, which you could imagine as a matmul of a 2x3 weight 矩阵 with an input activation of batch 1 and size 3. This is rotated compared to the previous slides and inputs flow out to the right instead of down, but you can roughly see the structure.

       We can efficiently pipeline this to multiply large matrices without too large a pipeline bubble. With that said, it’s important that our matrices have shapes larger than the side dimension of the MXU, which is generally 128x128. Some TPUs (since TPU v3) have multiple MXUs, either 2 for TPU v3 or 4 for TPU v4/5, so we need to ensure tiling dimensions are larger than 128 * number of MXUs. [Here’s](https://www.youtube.com/watch?v=sJltBQ4MOHA) a good animation for this.

 Trillium (TPU v6e) has a `256x256` systolic array, which means it can perform 4x more FLOPs / cycle. This also means the dimensions of your tensors need to be twice as large to utilize the MXU fully.

 [This blog post](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) has another excellent animation of a systolic array 乘法 for a fixed weight 矩阵.

      ### Miscellaneous *Work done at Google DeepMind, now at MatX.

   ### Citation For attribution in academic contexts, please cite this work as:

  ```
    Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.

```  or as a BibTeX entry:

  ```
    @article{scaling-book,
      title = {How to Scale Your Model},
      author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
      and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
      publisher = {Google DeepMind},
      howpublished = {Online},
      note = {Retrieved from https://jax-ml.github.io/scaling-book/},
      year = {2025}
    }

```       Please enable JavaScript to view the [comments powered by giscus.](http://giscus.app/?ref_noscript)      © Copyright 2026 . Powered by [Jekyll](https://jekyllrb.com/) with [al-folio](https://github.com/alshedivat/al-folio) theme. Hosted by [GitHub Pages](https://pages.github.com/).

---

## 📚 延伸阅读

1. [JAX官方文档](https://jax.readthedocs.io/)
2. [Transformer原始论文](https://arxiv.org/abs/1706.03762)
3. [TPU技术文档](https://cloud.google.com/tpu/docs)
4. [GPU并行计算指南](https://docs.nvidia.com/cuda/)

## 💡 学习建议

1. **实践结合**：在学习理论的同时进行代码实践
2. **循序渐进**：按照章节顺序逐步深入学习
3. **问题驱动**：带着实际问题学习效果更好
4. **社区交流**：参与相关技术社区讨论

---

*本翻译由OpenClaw自动生成，正在不断完善中。*
*如有翻译问题或建议，欢迎通过博客反馈渠道联系。*
