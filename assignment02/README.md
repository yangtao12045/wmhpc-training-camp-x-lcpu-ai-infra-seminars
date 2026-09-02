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
查询得
SM 数量 = 148
clocks.max.sm = 2032 MHz
8192*148*2.032*10e9约2464TFLOPS
由于clocks.max.sm不等同于Tensor Core频率，与datasheet存在误差
FP4差异较大，推测是由于NVFP4路径有额外的硬件吞吐加强

RTX5090约512FLOP/cycle/SM
512*170*2.407约209.5TFLOPS

FP4差异较大，也是因为RTX Blackwell Tensor Core对FP4有专门的增强

根据 bf16 峰值和显存带宽计算机器平衡点（FLOP/byte），并与单条 mma 的计算强度（S016，
m16n8k16 fp16 为 3.2 FLOP/byte）比较。思考两者之间的差距意味着什么，以及为什么后续 M2–M4
需要从数据供给路径入手优化。
机器平衡点较计算强度大近百倍，说明大部分时间花费在数据供给，所以需要从数据供给入手优化

```
| 量                      |                                  RTX 5090 |                                                         HGX B300 |
| ---------------------- | ----------------------------------------: | ---------------------------------------------------------------: |
| bf16 FLOP/cycle/SM |                                  512 |                                                         8192 |
| bf16 峰值            |                         209.5TFLOPS |                                                  2464TFLOPS|
| fp8 峰值（按位宽估）      |                         419TFLOPS |                                                  4927TFLOPS |
| fp4 峰值（按位宽估）       |                         838TFLOPS |                                                 9855TFLOPS |
| datasheet 对照       | BF16 209.5TF；FP8 419TF；FP4 1676TF | BF16 2250TF；FP8 4500TF；FP4 13500TF|
| HBM/GDDR 带宽        |                             1792GB/s |                                                    7700 GB/s |
| 机器平衡点（BF16）        |                         117FLOP/B |                                                 320 FLOP/B |


0.3 
```bash
(a) 一条 mma 的计算强度，分子是 2MNK，分母按 A、B 读入与 D 写回的字节总和计 (S016 的
口径)。
正确，这是计算强度计算公式
2(b) mma.sync 是 warp 级协作指令:32 个 lane 各持 fragment 的一部分，要求全 warp 一致地执行
这条指令；有 lane 发散时行为未定义。
正确
(c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度，而且没有代价，所以指令形状越大越
好。
错误，增大形状在一定程度能提高计算强度，但tile增大会有很多代价，所以不可能越大越好
(d) 只要单条 mma 的计算强度低于机器平衡点，GEMM kernel 就不可能逼近计算峰值。
错误，有多种优化数据供给的方法如pipeline，因此不能认为单条 mma 的计算强度决定kernel不能达到计算峰值

```

1.1 01_fragment_map.cu
```bash
附加问题（写入报告）：A 的同一个 b32 寄存器中的 4 个 fp8 元素沿矩阵哪个方向相邻？这个
布局对 1.4 中使用 ldmatrix load 有什么影响？
沿列方向连续，能高效运用ldmatrix load加载数据

```

1.2 02_bug_fragment.cu
```bash
(a) 描述症状：D 的哪些位置错、错成了什么 (和对的部分是什么关系)；
MISMATCH D[8][0]: got -1, want -11 MISMATCH D[8][1]: got -5, want 5 MISMATCH D[8][2]: got -16, want -14 MISMATCH D[8][3]: got 15, want 9
第8行的结果实际上是第0行的结果
(b) 修好它，并解释错的是哪个 fragment 的哪部分映射，为什么恰好产生 (a) 的症状。
错的是A矩阵的映射，行号映射错了，导致全部都是读取的上半矩阵

```

1.3 03_mma_fp8.cu


1.4 04_ldmatrix.cu 04_ldmatrix.ptx
```bash
(a) ldmatrix 省掉了手工装载中的哪些工作？
ld.shared.u8指令读取fp8与pack成mma目标fragment的指令，寄存器使用更少
(b) 为什么这些工作在手工装载路径中无法避免？
mma需要固定格式的fragment，fp8元素大小和mma register粒度不一致，普通load指令没有专门为fragment layout设计


```

1.5 
| 档位       | 预测 wavefront 比 | 实测 wavefront | 实测 conflict | 平均 cycle |
| -------- | -------------: | -----------: | ----------: | -------: |
| 32 B     |             2× |        16384 |        8192 |     9.72 |
| 64 B     |             4× |        32768 |       24576 |    10.75 |
| 128 B    |             8× |        65536 |       57344 |    16.06 |
| 128+16 B |             1× |         8192 |           0 |     9.23 |

```bash
比较预测与实测结果：哪一种行跨度使 wavefront 数增加到 4 倍？wavefront 的比例应与 bank
模型一致，但实际耗时的差距通常没有这么大。结合 8 个 warp 的占用情况，解释为什么 wavefront
增加 4 倍并不会使总耗时也增加 4 倍。
64B使wavefront数增加到4倍
因为occupancy足够高，内存延迟被上下文切换掩盖，所以没有线性增长


```


2.1
```bash
wgmma.mma_async
st.shared
wgmma.commit_group
fence.proxy.async
wgmma.fence
wgmma.wait_group

wgmma.fence确保warp group到达同步点
st.shared防止wgmma提前读取shm
fence.proxy.async避免读写shm跨代理乱序
wgmma.mma_async发射异步mma，计算访存重叠
wgmma.commit_group防止group管理与maa发射混乱
wgmma.wait_group防止maa完成前提前读取结果

(b) 判断下列说法是否正确，并给出一句理由。
1. fence.proxy.async 是 wgmma 专属的指令，TMA 与 tcgen05 的场景不需要它。
错，fence.proxy.async是proxy关键同步机制，TMA 与 tcgen05也需要使用
2. wgmma.commit_group 会阻塞，直到它之前发射的 wgmma 全部完成。
错，只提交，不会阻塞
3. 不加 fence.proxy.async 时，wgmma 可能读到 shared memory 中的旧值，因为 st.shared 的
写经过 generic proxy，而 wgmma 的读经过 async proxy。
对

```

2.2 02_descriptor.cu
```bash
场景 2 和场景 3 最终得到的 descriptor 相同。在报告中回答：MN-major 与 K-major 的区别 体现在哪里？
区别在b_major位表示，b_major=0表示K-major，b_major=1表示MN-major。

```

2.3 03_swizzle.cu

3.1
```bash
(a) tcgen05.ld 读取 TMEM 时，每个 warp 只能读取自己对应的 32 条 lane，warp 之间不能互相
读取。
对
(b) 与 mma.sync 由 warp 协作、wgmma 由 warpgroup 协作不同，tcgen05.mma 由单个线程发射，
随后由硬件异步执行。
对
(c) TMEM 中的累加结果可以直接通过 TMA 搬回 global memory，不需要经过寄存器。
错，accumulator需要通过寄存器才能搬回global memory
(d) TMEM 每个 SM 包含 128 lane × 512 column × 4 B；一个 m128n256 的 f32 accumulator 恰
好占用其中一半。
对，128*256*4=1/2*(128*256*4(sizeof(float)))
(e) tcgen05.commit 会阻塞直到之前发射的 mma 全部完成，因此 commit 返回后即可安全读取
TMEM
错，commit不会阻塞maa完成，需要等待追踪的mbarrier完成才能完全读取

```

3.2 02_single_tile.cu

4.1 01_tiled.cu

4.2 02_tma.cu

4.3 03_pipeline.cu
