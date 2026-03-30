---
title: "使用JAX编程TPU"
date: 2026-03-30
description: "使用JAX编程TPU - Programming TPUs in JAX"
categories:
  - JAX
  - 机器学习
  - 扩展计算
tags:
  - 实践教程
  - 翻译
---

# 使用JAX编程TPU

*Programming TPUs in JAX*

## 📋 章节概览

**所属部分**：实践教程
**原文标题**：Programming TPUs in JAX
**原文地址**：https://jax-ml.github.io/scaling-book/jax-stuff
**翻译时间**：2026年03月30日

## 🎯 本章要点

本章将深入探讨使用JAX编程TPU的相关内容，包括：

1. **核心概念**：理解使用JAX编程TPU的基本原理
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
Programming TPUs in JAX | How To Scale Your 模型     *                         [ How To Scale Your 模型 ](/scaling-book)  Toggle navigation     [](../profiling) [](../conclusion)  
  [ ](/scaling-book/) 
 * [Previous Part](../profiling)
 * [Next Part](../conclusion)
 *  [  Sections ](#)  [Part 0. Introduction](/scaling-book/index) [Part 1. Intro to Rooflines](/scaling-book/roofline) [Part 2. All About TPUs](/scaling-book/tpus) [Part 3. Sharded Matmuls](/scaling-book/分片) [Part 4. Transformers](/scaling-book/transformers) [Part 5. 训练](/scaling-book/训练) [Part 6. 训练 LLaMA](/scaling-book/applied-训练) [Part 7. 推理](/scaling-book/推理) [Part 8. Serving LLaMA](/scaling-book/applied-推理) [Part 9. Profiling](/scaling-book/profiling) [Part 10. All About JAX](/scaling-book/jax-stuff) [Part 11. Conclusions](/scaling-book/conclusion) [Part 12. GPUs](/scaling-book/gpus)  
 *       
 
            # Programming TPUs in JAX Part 10 of [How To Scale Your 模型](/scaling-book) ([Part 9: Profiling](../profiling) | [Part 11: Conclusions](../conclusion))

 How to use JAX to program TPUs efficiently! Much of this section is taken from [here](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html). You can run the code examples in this section with free TPUs on [Google Colab](https://colab.sandbox.google.com/).

      ### Contents  [How Does Parallelism Work in JAX?](#how-does-parallelism-work-in-jax)   [](#)  
 *  [Auto 分片 mode](#auto-分片-mode) 
 *  [“Explicit 分片 mode”](#explicit-分片-mode) 
 *  [Manual 分片 mode via shard_map](#manual-分片-mode-via-shard-map) 
 
  [Worked Problems](#worked-problems)    ## How Does Parallelism Work in JAX? JAX supports three schools of thought for multi-device programming:

 *  Compiler, take the wheel! Let the XLA compiler automatically 分区 arrays and decide what communication to add to facilitate a given program. This lets you take a program that runs on a single device and automatically run it on thousands without changing anything.
 *  JAX, take the wheel! Automatic parallelism is great, but sometimes the compiler does something crazy. Explicit 分片 lets you write single-device code like usual, but have JAX handle 分片 propagation (not the compiler). This means JAX can ask you for clarification when it’s unclear what you want.
 *  Just let me write what I mean, damnit! While compilers are nice, they sometimes do the wrong thing and add communication you don’t intend. Sometimes we want to be explicit about exactly what communication we intend to run.
 
    Mode View? Explicit 分片? Explicit Collectives?     Auto Global ❌ ❌   Explicit Global ✅ ❌   Manual Per-device ✅ ✅    Correspondingly, JAX provides APIs for each of these modes:

 *  `jax.jit` (with `Auto` mesh axes) lets you take any existing JAX function and call it with sharded inputs. JAX then uses XLA’s [Shardy](https://openxla.org/shardy) compiler which automatically parallelizes the program. XLA will add communication for you (AllGathers, ReduceScatters, AllReduces, etc.) when needed to facilitate existing operations. While it isn’t perfect, it usually does a decent job at automatically scaling your program to any number of chips without code changes.
 *  `jax.jit` with `Explicit` mesh axes looks similar to (1), but lets JAX handle the 分片 propagation instead of XLA. That means the 分片 of an array is actually part of the JAX type system, and JAX can error out when it detects ambiguous communication and lets the user resolve it.
 *  `jax.shard_map` is the more manual counterpart. You get a device-local view of the program and have to write any communication you want explicitly. Have a sharded array and want the whole thing on each device? Add a `jax.lax.all_gather`. Want to sum an array across your devices? Add a `jax.lax.psum` (an AllReduce). Programming is harder but far less likely to do something you don’t want.
 
 ### Auto 分片 mode `jax.jit` plays two roles inside JAX. As the name suggests, it “just-in-time” compiles a function from Python into bytecode (via XLA/HLO/LLO) so it runs faster. But if the input is sharded or the user specifies an `in_sharding` or `out_sharding`, it also lets XLA distribute the 计算 across multiple devices and add communication as needed. For example, here’s how you could write a sharded matmul using `jax.jit`:

 ```
import jax
import jax.numpy as jnp

# Running on a TPU v5e 4x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

# We create a 矩阵 W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df-&gt;bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)

``` This will run automatically with any 分片 and 分区 the 计算 across our devices. But what’s actually happening at the hardware level?

 * First we create In and W sharded across our devicesNotice how we did this. This is one way to create an array with a particular 分片 (i.e. by adding the device argument to the creation function). Another one is to create an array normally with `jnp.array(....)` and then do e.g. `jax.device_put(..., jax.P('x', 'y'))`. Yet another is to write a function which creates the array you want, and jit-compile it with `out_shardings` being what you want.. W is sharded 2-way along the contracting dimension, while In is sharded 4-way (along both the contracting and output dimensions). This corresponds to a 分片 W[DY, F] and In[BX, DY], aka a kind of 模型 and data parallelism.
 * If we were running this locally (i.e. on one device), `matmul_square` would simply square the input and perform a simple matmul. But because we specify the `out_shardings` as `P('X', None)`, the output will be sharded along the batch but replicated across the 模型 dimension and will require an AllReduce to compute.
 
 Using our notation from previous sections, this will likely do something like

 * Out[BX, F] { UY } = In[BX, DY] *D W[DY, F]
 * Out[BX, F] = AllReduce(Out[BX, F] { UY })
 
 `jax.jit` will add this for us automatically! We can actually print the HLO with `jit_matmul.as_text()` and see the following HLO (abbreviated dramatically):

 ```
# This fusion is the actual matmul of the sharded inputs and 矩阵
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)

``` We can see the matmul (the fusion) and the AllReduce above. Pay particular 注意力 to the shapes. `bf16[2, 1024]` is a local view of the activations, since our `batch_size=8` is split across 4 devices and our `d_model=2048` is likewise split 2 ways.

 This is pretty magical! No matter how complicated our program is, [Shardy](https://openxla.org/shardy) and jit will attempt to find shardings for all the intermediate activations and add communication as needed. With that said, Shardy has its flaws. It can make mistakes. Sometimes you’ll look at a profile and notice something has gone wrong. A giant AllGather takes up 80% of the profile, where it doesn’t need to. When this happens, we can try to correct the compiler by explicitly annotating intermediate tensors with `jax.lax.with_sharding_constraint`. For instance, with two matmuls I can force the intermediate activations to be sharded along the `y` dimension (not that this is a good idea) with the following:

 ```
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df-&gt;bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df-&gt;bd', hidden, Wout)

``` This makes up about 60% of JAX 并行 programming in the automatic partitioning world where you control the intermediate shardings via `jax.lax.with_sharding_constraint`. But “compiler tickling” is famously not a fun programming 模型. You could annotate every intermediate variable and still not know if you’ll get the right outcome. Instead, what if JAX itself could handle and control 分片 propagation?

 ### Explicit 分片 mode Explicit 分片 (or “分片 in types”) looks a lot like automatic 分片, but 分片 propagation happens at the JAX level! Each JAX 操作 has a 分片 rule that takes the shardings of the op’s arguments and produces a 分片 for the op’s result. You can see the resulting 分片 using `jax.typeof`:

 ```
import jax
import jax.numpy as jnp
import jax.分片 as shd
import numpy as np

# Running on a TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y]
  return out

f(x)

``` As you can see, JAX propagated the 分片 from input (`x`) to output (`x`) which are inspectable at trace-time via `jax.typeof`. For most operations these rules are simple and obvious because there’s only one reasonable choice (e.g. elementwise ops retain the same 分片). But for some operations it’s ambiguous how to shard the result in which case JAX throws a trace-time error and we ask the programmer to provide an `out_sharding` argument explicitly (e.g. jnp.einsum, jnp.reshape, etc). Let’s see another example where you have conflicts:

 ```
# We create a 矩阵 W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df-&gt;bf', jnp.square(In), W)

matmul_square(In, W)  # This will error

``` This code errors with `Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output 分片 via the `out_sharding` 参数. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)`

 This is awesome because how the output of einsum should be sharded is ambiguous. The output 分片 can be:

 * P(‘X’, ‘Y’) which will induce a reduce-scatter or
 * P(‘X’, None) which will induce an all-reduce
 
 Unlike Auto mode, explicit mode errors out when it detects ambiguous communication and requires the users to resolve it. So here you can do:

 ```
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df-&gt;bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]

``` Auto mode and Explicit mode can be composed via `jax.分片.auto_axes` and `jax.分片.explicit_axes` APIs. This is a [great doc to read](https://docs.jax.dev/en/latest/notebooks/explicit-分片.html) for more information.

 ### shard_map: explicit parallelism control over a program While Shardy is the “compiler take the wheel” mode, jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) puts everything in your hands. You specify the 分片 of the inputs, like in jax.jit, but then you write all communication explicitly. Whereas `jax.jit` leaves you with a global cross-device view of the program, `shard_map` gives you a local per-device view.

 Here’s an example. Try to reason about what this function does:If you want to play with this yourself in a colab by emulating a mesh, you can do so using the following cell `import jax; jax.config.update('jax_num_cpu_devices', 8)`

 ```
import jax
import jax.numpy as jnp
import jax.分片 as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)

``` What does this do? `slice_and_average` is run on each TPU with 1/8th of the array, from which we slice the first 4 elements and average them across the full mesh. This means we’re effectively doing `mean(x[:4], x[64:68], x[128:132], …)`. This is pretty cool, because that’s not an easy 操作 to express in JAX otherwise.

 Why do this instead of jax.jit? If we’d used `jax.jit`, `slice_and_average` would have seen a global view of the array (the full `[512,]` array). We’d have had to slice out this non-uniform slice and then perform an average which XLA would have had to interpret correctly. XLA might have added the wrong communication or gotten confused. Here we see the local view and write only the communication we need.

 Example [Collective Matmul]: To take a more realistic example, say we want to implement 模型 parallelism where the activations are initially 模型 sharded, i.e. A[BX, DY] * W[D, FY] -&gt; Out[BX, FY]. Naively, we would do this by AllGathering A first followed by a local 矩阵 乘法:

 * A[BX, D] = AllGatherY(A[BX, DY])
 * Out[BX, FY] = A[BX, D] *D W[D, FY]
 
 Sadly, this is bad because it doesn’t allow us to overlap the communication with the 计算. Overlapping them can be done with a “collective matmul”, as described in [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959). The 算法 is basically as follows:

 * For each Y shard, perform a matmul of the local chunk of A with the local chunk of W, producing a result of shape `[B / X, F / Y]`. Simultaneously, permute A so you get the next chunk locally, perform the matmul, and sum the result.
 
 We can implement that quite easily with `jax.shard_map`:

 ```
import functools

import jax
import jax.numpy as jnp
import jax.分片 as shd
import numpy as np

# This is intended to run on a TPU v5e-8 runtime. If you can't get this,
# try setting jax.config.update('jax_num_cpu_devices', 8).
#
mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)

``` This is pretty neat! We can benchmark this and see that it’s also a lot faster! [Here’s](https://imgur.com/a/e9I6SrM) the profile with the default jit matmul which takes 311us with a big blocking AllGather at the beginning:

       And [here’s](https://imgur.com/a/21iy0Sv) the version above that takes 244 us. You can see the profile doesn’t have the AllGather. It’s all useful work! Our FLOPs utilization is also a lot higher.

       It’s also worth noting that the matmul time with no 分片 on the contracting dimension is [224us](https://imgur.com/a/i3gNKfq), so we’re remarkably close to the unsharded baseline here. This is a good example of the kind of 性能 engineering you might end up doing to improve TPU utilization. For more `shard_map` examples, [this note is great](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

 Now here are a couple of useful worked problems to try and implement using `jax.jit` or `shard_map`!

 ## Worked Problems Here are some random JAX-related problems. I’ll add some more later. For all of these, you’ll need some number of TPUs in a Colab. You can use a public Colab with TPUv2-8. From now on, we’ll assume you have N devices available.

 Problem 1: Let A be an array of activations of shape float32[SX, DY] with `X * Y = N`. Do the following:

  Write a function in JAX that computes the average within each `(X, Y)` shard, i.e. it returns an array of size [X, Y] where `arr[i, j]` is the average over shard `(i, j)`. Do this with both `jax.jit` and `shard_map`. Profile each and see how long they took. Was there any communication added? Hint: there shouldn’t be, but sometimes XLA adds it anyway.

   Write a function in JAX that returns roll(x, shift, axis=0) - x for some shift within each shard X. I’m not enough of a masochist to make you do this in jax.jit, so just do this with `shard_map`.

 Click here for the answer. Part 1: Here is a solution to part 1. Note the fairly complex reshapes we have to do for the `jax.jit` solution.

 ```
import numpy as np

import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)

``` Part 2: Here is a similar solution to Part 2.

 ```
import numpy as np

import jax
import jax.numpy as jnp

import functools

P = jax.分片.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)

```  Problem 2: Here we’ll make a basic “mixture of experts” 模型 together. Let W: float32[EX, D, F] be a set of E “expert” matrices. Let A: float32[SX, D] (our activations) and let B: int32[SX] be a set of “routing assignments” where B[i] is an integer in the range `[0, E)` telling us which 矩阵 we want to process that 激活. We want to write a function in JAX that returns `Out[i] = W[B[i]] @ A[i]`.

  Let’s start by ignoring 分片 altogether. Make all of these tensors small enough so they fit in one device. Write a local implementation of this function. Make sure you don’t materialize an array of shape `[S, D, F]`! Hint: try sorting the tokens into a new buffer of shape `[E, S, D]` with some 注意力 to masking (why do we need the second dimension to have size S?).

   If you just `jax.jit` the above method, something will happen. Profile this and see what communication it decided to do. How long does it take?

   One problem you’ll notice with the above is that it likely gathers the full set of activations A locally, i.e. AllGatherX([SX, D]). Not only is this expensive communication-wise, it’s also incredibly expensive 内存-wise if we can’t fit the full set of activations locally. Implement the above using `shard_map` and explicit communication.

  For a first pass, it might be easiest to use a `jax.lax.all_gather` and reorder as in (a).

   For a second pass, try to avoid materializing any array of size `[E, S, D]`, i.e. try to perform the 计算 in a ragged fashion using a `jax.lax.all_to_all` inside a `jax.lax.while_loop`. This way, you can avoid materializing the full activations and wasting compute on padding. How much faster is this than your original implementation?

   Most MoEs route to multiple (k) experts and then average the result. Refactor the above to implement this. Let B: int32[S, k] in this case for the k experts to route to.

 Click here for the (partial) answer. 1/2. For part (1), you have a lot of choices. Here’s one option that just iterates over the experts with masking.

 ```
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -&gt; jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = lax.scan(expert_forward, output, jnp.arange(E))

    return output

``` You can also use `jax.lax.ragged_dot` which will do something similar but more efficiently.

 * I’m only going to sketch the pseudocode here (if you have a clean solution feel free to add it):
 
 ```
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size].any()):
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
  return concat(out)

``` The basic idea is to iterate over chunks of the array, sort them and do an all_to_all, then do the local FLOPs.

  Problem 3: The collective matmul example above is actually super relevant for real LLMs. Let’s tweak the example to do the full Transformer stack.

  As an exercise, let’s start by implementing an AllReduce collective matmul, i.e. A[BX, DY] *D W[DY, F] -&gt; Out[BX, F]. Note that the output isn’t replicated. The naive 算法 is discussed above, basically just a local matmul followed by an AllReduce. Try to make a comms overlapped “collective” version of this 操作. Hint: tile over the output dimension and feel free to use `jax.lax.psum` (aka AllReduce). Note: due to the way XLA handles this, it may not actually be faster than the baseline.

   The complement to the AllReduce collective matmul above is a ReduceScatter collective matmul, as in Tmp[BX, FY] *F W2[FY, D] -&gt; Out[BX, DY]. This occurs in the down-projection 矩阵 in a Transformer. Implement a collective, overlapped version of this in JAX. Be careful about passing only the minimal amount of data you need. Hint: try permuting the result as you accumulate it.

   Put these two together into an end-to-end Transformer block that performs In[BX, DY] *D Win[D, FY] *F Wout[FY, D] -&gt; Out[BX, DY] with overlapped communication.As before, we can't do $W_{in} \cdot W_{out}$ first because of a non-linearity we've omitted here. How much faster is this than a `jax.jit` implementation?

 Problem 4: All of the collective matmuls implemented above are unidirectional: they only permute in one direction. Rewrite the collective AllReduce matmul and the collective ReduceScatter matmuls to use bidirectional communication. How much faster are these?

 ### That’s all for Part 10. That’s basically it! For final conclusions and further reading, click [here](../conclusion).      ### Miscellaneous *Work done at Google DeepMind, now at MatX.

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
