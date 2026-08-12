# Weiming HPC Training Camp x LCPU AI Infra Seminars

北京大学未名超算队（The Radiance of Weiming）与北京大学学生 Linux 俱乐部（LCPU）合办的暑期 AI Infra 系列活动仓库。每次 session 的作业与配套代码放在对应的 `assignmentXX/` 目录。

[official website](https://infra.seminars.lcpu.dev)

## 内容

系列从 GPU 编程起步，此后依次进入 memory hierarchy、Tensor Core、软件流水、DSL 与编译器、通信、推理系统、RL 系统。完整时间表以活动推送为准。

| 目录 | Session | 主题 |
| --- | --- | --- |
| [assignment01](assignment01/) | 1（7.26） | GPU & GPU Programming——SIMD/SIMT、CUDA 编程模型、Triton/TileLang |

## 使用方式

每个 assignment 目录自带 README、handout PDF 与基础代码。

## 关于 AI 使用

policy 见 [CLAUDE.md](CLAUDE.md)（[AGENTS.md](AGENTS.md) 内容相同，供其他工具读取）。AI 可以帮你理解，但不能替你实现。

0.2
```bash
GPU 型号            : NVIDIA GeForce RTX 5090
compute capability  : 12.0
SM 数量             : 170
warp 大小           : 32
shared mem / block  : 49152
max threads / SM    : 1536
global mem          : 33668857856
max threads / block : 1024
```

1.1
```bash
(a) 一块标称 100 TFLOPS 的 GPU，执行单条指令的延迟一定低于5GHz的CPU。
错，100Tflops表示计算吞吐能力，与单条指令的延迟没有必然联系，无法与5GHzCPU比较单条指令延迟
(b) HBM 的“高带宽”指大块连续访问时的吞吐，零散的随机访问达不到标称值。
对
(c) 严格串行的迭代算法（每步依赖上一步的结果），即使换一块算力更强的GPU也快不了多少。
对，gpu主要优势在于能并行的高度密集计算，在严格串行方面没有特别优势
(d)“算力 1000 TFLOPS”意味着每次运算的延迟是 10−15 秒。
错，与a相似，tflops指的是吞吐能力，并非反映单次运算能力
```

1.2
```bash
Session 1 讲座里提过“N 方过百万”这个例子。总计算量1012 FLOP在当代GPU上的运算时间
大概是毫秒级，那为什么一个严格在线的串行算法仍然做不到几秒内跑完？（从“延迟”和“吞吐”
的角度考虑）
gpu强大的计算能力源于其大规模并行的高吞吐，严格串行程序无法发挥gpu优势，而gpu对单次计算的延迟较高且在串行背景下无法被隐藏，所以做不到几秒内跑完
```


1.3
| 执行层次            | 软件含义                              | 对应硬件                                        | 直接可用的存储                             | 同步与通信手段                                                                                 |
| --------------- | --------------------------------- | ------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| **thread**      | kernel 的最小执行单位                    | 计算单元上的一个 lane                               | 自己的寄存器                              | 自身天然有序                                                                                  |
| **warp**        | 32个thread组成的单元   | SM由scheduler调度的32个lanes        | 各thread的寄存器 | __syncwarp()，shuffle/vote等primitive                                    |
| **block / CTA** | 一组threads，由若干warp组成     | 整个block驻留在SM上 | shared memory  | __syncthreads()，shared memory，atomics                |
| **grid**        | 全部 blocks的集合 | 分布到多个SM                        | global memory    | global memory，atomics通信 |

1.4
```bash
SIMD 与 SIMT 的区别？另：判断正误——Nvidia GPU 在 Volta 之后每个线程有独立的program
counter，所以 branch divergence 不再有性能代价。
SIMD与向量化指令关系密切，可理解为一条指令操作多个数据元素，SIMT是多个独立线程处理不同数据。SIMD基本单位是向量而SIMT基本单位是线程，SIMT由于本质是线程，允许不同控制流，SIMD通常不能很好地处理不同控制流。
错，在wrap内一般用掩码处理分支，不同分支不能完全利用资源，branch divergence仍有很大代价
```

1.5
| 配置                             |   耗时 (ms) |  ns / 元素 |
| ------------------------------ | --------: | -------: |
| CPU 单线程                        |     9.875 |     2.35 |
| GPU `<<<1, 1>>>`               |   136.604 |    32.57 |
| GPU `<<<1, 256>>>`             |     2.327 |     0.55 |
| GPU 铺满 grid `<<<16384, 256>>>` | **0.024** | **0.01** |
```bash
回答：(a)GPU单线程为什么比CPU慢这么多？(b)从单block到铺满
grid 的提速，说明 GPU 加速计算靠的是什么？
(a)cpu通常有更强的单线程能力，单线程延迟显著低于gpu
(b)gpu的能力主要靠大规模计算的高吞吐隐藏延迟
```

2.2
```bash
为下列五个场景选择正确的修饰符（如__global__等）。
(a) 在 GPU 上执行、由CPU侧启动的kernel 函数。
__global__
(b) 只会被 kernel 调用的辅助函数。
__device__
(c) host 和 device 代码都要调用的小工具函数。
__host__ __device__
(d) 整个 kernel 运行期间不变、所有线程都要读的系数表。
__constant__
(e) block 内线程共享的暂存数组。
__shared__
```

2.3
```bash
搬运 + kernel + 读回: 106.4 ms
搬运 + kernel + 读回: 95.3 ms
：(a)kernel 启动之后、CPU读结果之前，为什么必须有一次同步？在原
先的版本里这次同步发生在哪个调用里？
cpu在kernel launch后会不阻塞继续执行接下来的指令，因此必须同步确保kernel已完成
(b)对比两版“搬运+kernel+读回”的耗时，分析差距
的原因（谁快谁慢都有可能，与使用的卡有关）。
Unified Memory更快，可能是按需迁移页面的开销比显式管理更小
```

2.4
```bash
(a) vectorAdd<<<...>>>(...) 这条语句返回时，kernel 一定已经执行完毕。
错，cpu launch后不会阻塞，会继续执行后续指令
(b) 同一个 stream 里，cudaMemcpy（device 到 host）会等它前面的 kernel 全部完成后才开始拷
贝。
(c) kernel 内部的非法访存，会在启动语句处同步地报出来。
错，异步的，不会同步报出来
```

2.5
```bash
bug在于每个block线程超上限
```

2.7
```bash
然后请回答——这种写法的价值在哪里？launch 只有 16384 个线程时，性能上要付出什么代
价？
价值在于无论n多大kernel都能正常运行，提高线程复用，代价是并行能力降低，可能无法充分利用计算资源
```

2.8
```bash
(a) 顺序由谁决定？
由调度器决定，不能保证block的执行顺序
(b)程序的正确性可以依赖block的执行顺序吗？这条限制和Guide1.1说
的scalable programming model 有什么关系？
不可以，必须block间无依赖。这样的编程前提正是scalable programming model的基础
```
