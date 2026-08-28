# C1:FlashKDA——官方 kernel 停在 SM80 MMA

## 背景

Kimi K3 的线性注意力部分是 KDA(Kimi Delta Attention)。Moonshot 开源的
FlashKDA(本目录 `FlashKDA/` 快照)是其高性能 forward kernel,在 GB200 上
对 fla 的 Triton `chunk_kda` 有 1.7-3.3 倍加速(`BENCHMARK_GB200.md`)。
值得注意的是:这份 kernel 的矩阵乘用的是 SM80 世代的 `mma.sync`,而不是
wgmma(SM90)或 tcgen05(SM100)——在 GB200/B300 上运行时同样如此。
作者在 `docs/20260420-flashkda-v1-deep-dive.md` 里解释了设计,但"为什么
停在 SM80 指令"这个决策的量化论证是留白的。这道题就是把这份论证做出来
——或者推翻它。

## 任务(三层)

1. 复现:在 B300 上装起 FlashKDA,跑通官方 benchmark(`benchmarks/`,
   形状对照 `BENCHMARK_GB200.md`),用 ncu/SASS 确认计算主路径确实是
   SM80 MMA(`benchmarks/ncu.sh` 是官方的 ncu 模板)。
2. 分析:下面的讨论点逐个给出"结论 + 证据"。量化类的先纸面推算,
   再用 microbench 验证。
3. 挑战:选 SM100 路线的任一切面动手——只换指令不动算法、大 CHUNK +
   rescale、并行度重构,任选其一。正确性对 `fla_kda_ref/` 的实现对拍
   (`naive.py` 是纯 PyTorch 朴素参考,`chunk.py` 是 Triton 参照),
   性能对 FlashKDA 本体。做不出正收益也算完成:把"官方停在 SM80 是
   对的"论证扎实,就是讨论点 6 的另一半答案。

交付:代码 + 报告 + 答辩。

## 讨论点

1. CHUNK=16 的三个理由——bf16 数值范围、16×16 Neumann 级数求逆、
   SM80 MMA 形状匹配——各自量化:CHUNK=32/64 时哪个先破,代价多大?
2. tcgen05 的最小 tile 与 CHUNK=16 的形状匹配吗?不动 CHUNK 只换指令
   有没有收益——先纸上算,再 microbench 验证。
3. 递推在 chunk 间有状态依赖,并行度还能从哪来(多 head 进一个 CTA /
   persistent kernel / 2-CTA)?列出候选方案,互相找反例。
4. 这个负载在你们的卡上是 compute-bound 还是 memory-bound?用哪几个
   ncu metric 回答?(assignment 4.5 的瘦 GEMM 表是现成的参照系,
   `in_proj_qkvgfab` 就是 KDA 的输入投影。)
5. 状态存 bf16 的精度验证怎么设计?官方只说内部测试通过,拿出你们的
   验证方案和数据。
6. 假设你们是作者:v2 出不出 sm100a 专版?把可移植性与峰值两边的论据
   都写全,给出你们的结论。

## 材料

- `FlashKDA/`:官方仓库快照,pin commit `1ce47ea`(2026-07-29)。
  cutlass 子模块未含在快照里,构建时用
  `git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA`
  后 `git checkout 1ce47ea`(cutlass pin `5c149f5`)。
  重点文件:`docs/20260420-flashkda-v1-deep-dive.md`(设计文档)、
  `BENCHMARK_GB200.md`(官方数据表)、`csrc/smxx/`(kernel 本体)、
  `benchmarks/`(bench 与 ncu 脚本)、README(chunk_kda 调用约定与
  dispatch 调试方法)。
- `fla_kda_ref/`:fla-org/flash-linear-attention 的 `fla/ops/kda/` 快照,
  pin commit `a3edffc`。`naive.py` 纯 PyTorch 参考;`chunk.py` 及其依赖
  是 Triton 参照;`backends/flash_kda.py` 是 fla 调用 FlashKDA 的适配层
  (两边张量约定的对照表)。
- 形状:K3 的 KDA 配置是 96 头 × head_dim 128(93 层中 69 层 KDA、
  24 层全注意力);官方 benchmark 的 `T=8192, H=96, D=128` 即此,
  H=64 组只是附加对照形状。注意 TP 部署下每卡头数是 96/TP(TP8 为
  12)——讨论并行度时用每卡数;GEMM 侧形状见 assignment 4.5。

```bash
  flash_kda (bf16 state) : mean=1.7469 ms, min=1.7413 ms, max=1.8456 ms
  flash_kda (no state)   : mean=1.7525 ms, min=1.7475 ms, max=1.7960 ms
  flash_kda (fp32 state) : mean=1.7115 ms, min=1.6994 ms, max=3.7646 ms
  chunk_kda : mean=3.5656 ms, min=3.5446 ms, max=5.7096 ms
  chunk_gated_delta_rule : mean=1.9585 ms, min=1.9480 ms, max=2.1401 ms
varlen shape=[8192,96,128] seq_lens=[1300, 547, 2048, 963, 271, 3063] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.4734 ms, min=1.4035 ms, max=1.6886 ms
  flash_kda (no state)   : mean=1.4734 ms, min=1.3933 ms, max=1.5633 ms
  flash_kda (fp32 state) : mean=1.4914 ms, min=1.4219 ms, max=1.7438 ms
  chunk_kda : mean=3.6618 ms, min=3.6388 ms, max=5.7584 ms
  chunk_gated_delta_rule : mean=1.9529 ms, min=1.9440 ms, max=2.1179 ms
varlen shape=[8192,96,128] seq_lens=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.1887 ms, min=1.1810 ms, max=3.2783 ms
  flash_kda (no state)   : mean=1.1840 ms, min=1.1780 ms, max=3.1566 ms
  flash_kda (fp32 state) : mean=1.2115 ms, min=1.2069 ms, max=1.3951 ms
  chunk_kda : mean=3.5601 ms, min=3.5446 ms, max=5.6059 ms
  chunk_gated_delta_rule : mean=1.8609 ms, min=1.8499 ms, max=2.0170 ms
shape=[8192,64,128] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.5971 ms, min=1.5919 ms, max=1.7206 ms
  flash_kda (no state)   : mean=1.6043 ms, min=1.5998 ms, max=1.6693 ms
  flash_kda (fp32 state) : mean=1.5527 ms, min=1.5486 ms, max=1.6779 ms
  chunk_kda : mean=2.4322 ms, min=2.4212 ms, max=2.6233 ms
  chunk_gated_delta_rule : mean=1.3334 ms, min=1.3162 ms, max=3.4555 ms
varlen shape=[8192,64,128] seq_lens=[1300, 547, 2048, 963, 271, 3063] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.1149 ms, min=1.1063 ms, max=3.1525 ms
  flash_kda (no state)   : mean=1.1125 ms, min=1.1066 ms, max=1.1823 ms
  flash_kda (fp32 state) : mean=1.1237 ms, min=1.1166 ms, max=1.1779 ms
  chunk_kda : mean=2.5693 ms, min=2.5495 ms, max=3.2710 ms
  chunk_gated_delta_rule : mean=1.4233 ms, min=1.4127 ms, max=2.0525 ms
varlen shape=[8192,64,128] seq_lens=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=0.8022 ms, min=0.7985 ms, max=1.0428 ms
  flash_kda (no state)   : mean=0.7998 ms, min=0.7962 ms, max=1.0348 ms
  flash_kda (fp32 state) : mean=0.8201 ms, min=0.8156 ms, max=1.0683 ms
  chunk_kda : mean=2.3673 ms, min=2.3506 ms, max=2.5848 ms
  chunk_gated_delta_rule : mean=1.2553 ms, min=1.2467 ms, max=1.6876 ms
```

```bash

    Duration                         us       272.90
          Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512
          affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which
    Block Size                                                   256
    Grid Size                                                  49152
    Waves Per SM                                               41.51
    Section: Occupancy
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        96.54
    Duration                         us       748.45
    Block Size                                                   192
    Grid Size                                                     96
    Waves Per SM                                                0.32
          concurrently with other workloads, consider reducing the block size to have at least one block per
    Section: Occupancy
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Theoretical Occupancy                     %        18.75
    Achieved Occupancy                        %         9.37
          The 3.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
          hardware maximum of 16. This kernel's theoretical occupancy (18.8%) is limited by the required amount of
```
| CHUNK |  gate | decay 首次 zero | restore 首次 inf | 判断                  |
| ----: | ----: | ------------: | -------------: | ------------------- |
|    16 |    -5 |             无 |              无 | 最坏 lower bound 下仍安全 |
|    16 | -4/-2 |             无 |              无 | 安全                  |
|    32 |    -5 |            18 |             18 | 严重失败                |
|    32 |    -4 |            22 |             23 | 失败                  |
|    32 |    -2 |             无 |              无 | 此分布下安全              |
|    64 |    -5 |            18 |             18 | 严重失败                |
|    64 |    -4 |            22 |             23 | 严重失败                |
|    64 |    -2 |            44 |             45 | 中等负 gate 也失败        |


```bash
Single-warp
164.456 ns/inverse
6.081 million inverse/s

这是单 CTA、单 warp、重复调用后的 steady-state 延迟代理。它摊薄了 global memory 和 kernel launch，但各次 inverse 没有跨 iteration 数据依赖，因此不要称为严格的 dependent latency。

Saturated throughput
7.680 billion inverse/s
0.130206 ns/inverse（全 GPU reciprocal throughput）
```
