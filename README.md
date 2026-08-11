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
