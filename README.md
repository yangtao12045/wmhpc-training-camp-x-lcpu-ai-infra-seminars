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
(a) 一块标称 100 TFLOPS 的 GPU，执行单条指令的延迟一定低于5GHz的CPU。
错，100Tflops表示计算吞吐能力，与单条指令的延迟没有必然联系，无法与5GHzCPU比较单条指令延迟
(b) HBM 的“高带宽”指大块连续访问时的吞吐，零散的随机访问达不到标称值。
对
(c) 严格串行的迭代算法（每步依赖上一步的结果），即使换一块算力更强的GPU也快不了多少。
对，gpu主要优势在于能并行的高度密集计算，在严格串行方面没有特别优势
(d)“算力 1000 TFLOPS”意味着每次运算的延迟是 10−15 秒。
错，与a相似，tflops指的是吞吐能力，并非反映单次运算能力
