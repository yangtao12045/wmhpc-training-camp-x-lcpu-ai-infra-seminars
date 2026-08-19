# Assignment 02 · Tensor Core & Pipeline

Session 3（Tensor Core：从 mma.sync 到 tcgen05）与 Session 4（tiling、TMA、
pipeline）的配套作业。具体题目见 `handout/assignment02.pdf`；PDF 由
`handout/src/assignment02.md` 生成，以 Markdown 源文件为准。

## 目录结构

```
handout/assignment02.pdf  作业本体 START HERE
cuda/                 模块 0-5 的 CUDA 练习
  common.h            错误检查、计时、对拍的公共工具
  Makefile            编译(注意 ARCH 的用法,见下)
  m5_lowprec/         模块 5:低精度与 block scaling
kernels/              模块 5 的 Python 练习(5.1 / 5.2)
tests/                pytest 判测
team/                 团队选做题 C1 / C2 的材料与任务书
```

## 硬件

- 模块 0-1 在 RTX 5090 或 B300 上可完成
- 模块 2 全部无卡可判
- 模块 3-4 以及 5.3-5.4 需要 B300（sm_100 家族）；5.1-5.2 是 host
  Python 实验，不需要 GPU

- 各题的卡要求在题面标注

## 构建

CUDA 部分统一用显式 `-gencode`,Makefile 已配好,默认 `ARCH=100f`
(覆盖 B200/B300 全家族):

```bash
cd assignment02/cuda
make run/m5_lowprec/03a_encode_check      # 编译并运行单个练习
ARCH=89 make bin/...                      # 老架构按需指定
```

注意:`nvcc -arch=sm_103a` 这类简写在 CUDA 13.0 下会把用户代码的 PTX
target 展开成不带后缀的 compute_103,arch-specific 指令(例如 e2m1 的
cvt)会被 ptxas 拒绝,报错信息还会误导为"指令不支持"。显式写
`-gencode arch=compute_100f,code=sm_100f` 没有这个问题。这就是 Makefile
不用 `-arch` 简写的原因。

## Python 部分

```bash
cd assignment02
uv sync && uv run pytest tests/
```

M6 另外使用固定的 TileLang 版本：

```bash
uv sync --extra tilelang
```

## 关于 AI

必做题沿用系列惯例(仓库根目录 CLAUDE.md):AI 可以帮你理解、review、
解读报错,但每道题的实测数据必须来自你自己跑的卡,解释必须是你自己
能答辩的。团队题(C1/C2)不设限制,怎么用 AI 都可以;答辩时问的是
你们的决策和证据,答不上来的部分不算数。

0.1
```bash
nvcc -O2 -std=c++17 -I. --expt-relaxed-constexpr -gencode arch=compute_100f,code=sm_100f -o bin/m0_env/01_first_mma m0_env/01_first_mma.cu ./bin/m0_env/01_first_mma D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2 PASS rm bin/m0_env/01_first_mma

ARCH=120a make -B run/m0_env/01_first_mma nvcc -O2 -std=c++17 -I. --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a -o bin/m0_env/01_first_mma m0_env/01_first_mma.cu ./bin/m0_env/01_first_mma CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:76: no kernel image is available for execution on the device make: *** [Makefile:42: run/m0_env/01_first_mma] Error 1 rm bin/m0_env/01_first_mma

尝试用ARCH=120a时编译成功却运行失败。这说明生成了sm120a的机器代码，因此运行失败
```

0.2
```bash
采用dense，boost频率，FMA记作2FLOP
B300每个SM有4个Tensor Core，每个 Tensor Core每周期做1024个FMA，2*4*1024=8192FLOP/cycle/SM

```
| 量                      |                                  RTX 5090 |                                                         HGX B300 |
| ---------------------- | ----------------------------------------: | ---------------------------------------------------------------: |
| **bf16 FLOP/cycle/SM** |                                  **1024** |                                                         **8192** |
| **bf16 峰值**            |                          **419.5 TFLOPS** |                                                  **2250 TFLOPS** |
| **fp8 峰值（按位宽估）**       |                          **839.1 TFLOPS** |                                                  **4500 TFLOPS** |
| **fp4 峰值（按位宽估）**       |                         **1678.1 TFLOPS** |                                                  **9000 TFLOPS** |
| **datasheet 对照**       | 3352 AI TOPS ≈ sparse FP4，约为 dense FP4 ×2 | BF16 dense≈2.25 PF；FP8 dense≈4.5 PF；FP4 dense=14 PF，FP4 高于简单位宽估计 |
| **HBM/GDDR 带宽**        |                             **1792 GB/s** |                                                    **7700 GB/s** |
| **机器平衡点（BF16）**        |                          **234.1 FLOP/B** |                                                 **292.2 FLOP/B** |


