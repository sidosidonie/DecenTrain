# Multi-Machine 传输开销 vs GPU 计算开销对比

> attn-llama 和 FFN 两层在多机分布式验证场景下，单 round 传输的数据形状/字节、GPU forward 耗时，以及在不同带宽下的 wire 时间 vs GPU 时间对比。
> 数据日期：2026-05-15
> 关联文档：[MULTI_MACHINE.md](MULTI_MACHINE.md), [MULTI_MACHINE_FFN_REPORT.md](MULTI_MACHINE_FFN_REPORT.md)

---

## 0. 两种 attn 协议模式

attn-llama 支持两种 wire 协议（通过 `--send-scores` 切换）：

| 模式 | wire 上的 tensor | wire 字节 | coord 端处理 |
|---|---|---|---|
| **A. linear-only**（默认） | Q, K, V, O | **16.78 MB** | CPU 重算整套 attn (RoPE + qk^T + softmax + probs@v)，再 SLALOM-verify O |
| **B. with-scores** (`--send-scores`) | Q, K, V, **scores**, O | **33.55 MB** | 用收到的 q/k 重算 scores 做 MSE 校验；后续 softmax + probs@v 仍在 coord 跑 |

模式 B **不省 coord CPU**（仍需重算 qk^T 来对比），只是把 worker 的 qk^T 输出搬到 coord 验证一次。要真正省 CPU 需要换成 Freivalds 风格验证（左为 future work）。

---

## 1. 测试配置

| 项 | 值 |
|---|---|
| Wire dtype | fp16 |
| batch | 1 |
| seq | 512 |
| hidden | 4096 |
| 测试方向 | worker → coord（GPU 算完把激活回传给 CPU verify） |
| 协议 | "输出仅传"（只传 linear/attn 输出，输入和权重两端用 seed 同步生成） |

attn-llama 用 LLaMA 风格 multi-head attention：heads=32, kv_heads=32, head_dim=128。
FFN 用 LLaMA 风格 SwiGLU MLP：inter=11008。

---

## 2. 单 round 传输数据量

### 2.1 attn-llama (默认模式 A: linear-only)

worker 算完 attention forward，回传 4 个 fp16 tensor 给 coordinator：

| Tensor | 含义 | shape | numel | bytes (fp16) |
|---|---|---|---|---|
| Q | q_proj 输出 | [1, 512, 4096] | 2,097,152 | 4,194,304 (4.00 MiB) |
| K | k_proj 输出 | [1, 512, 4096] | 2,097,152 | 4,194,304 (4.00 MiB) |
| V | v_proj 输出 | [1, 512, 4096] | 2,097,152 | 4,194,304 (4.00 MiB) |
| O | o_proj 输出（attn_out 投影后） | [1, 512, 4096] | 2,097,152 | 4,194,304 (4.00 MiB) |
| **payload 总计** | | | **8,388,608** | **16,777,216 B = 16.00 MiB** |
| 加 framing 头（4×32B + 24B） | | | | +152 B |
| **每 round wire** | | | | **16,777,368 B ≈ 16.78 MB ≈ 134.22 Mbit** |

注：coordinator 自己 CPU 上从 q/k/v 重算 attn_out（带 RoPE + softmax + probs@v），所以 worker 只需传 q/k/v/o **四个 linear 输出**，不传 attn_out 本身。

### 2.1.b attn-llama (模式 B: with-scores)

打开 `--send-scores` 后，worker 额外传 **post-RoPE qk^T scores** 给 coordinator 做对比验证：

| Tensor | 含义 | shape | numel | bytes (fp16) |
|---|---|---|---|---|
| Q, K, V, O | 同 mode A | — | 8,388,608 | 16,777,216 (16.00 MiB) |
| **scores** | post-RoPE qk^T (pre-mask, pre-softmax) | **[1, 32, 512, 512]** | **8,388,608** | **16,777,216 (16.00 MiB)** |
| **payload 总计** | | | **16,777,216** | **33,554,432 B = 32.00 MiB** |
| 加 framing 头（4×32B + 36B + 24B） | | | | +188 B |
| **每 round wire** | | | | **33,554,620 B ≈ 33.55 MB ≈ 268.44 Mbit** |

**注意 scores 跟 Q/K/V/O 加起来一样大**（每个都是 16 MiB） —— 因为 attention scores 是 [B, heads, S, S] 的，S=512 时 S² = 262K 对每个 head 都要存。这就是为什么"也传 qk^T"会让 wire 翻倍。

实测 (2026-05-15 loopback, hidden=4096 seq=512 fp16)：
- 模式 A: 16,777,368 B / round
- 模式 B: 33,554,620 B / round（**+100% wire**）
- scores MSE：~7e-7（远低于 1e-2 阈值，校验稳定通过）

### 2.2 FFN

worker 算完 SwiGLU forward (`y2 = w2(silu(w1(x)) * w3(x))`)，回传 3 个 fp16 tensor：

| Tensor | 含义 | shape | numel | bytes (fp16) |
|---|---|---|---|---|
| y1 | w1 (gate proj) 输出 | [1, 512, 11008] | 5,636,096 | 11,272,192 (10.75 MiB) |
| y3 | w3 (up proj) 输出 | [1, 512, 11008] | 5,636,096 | 11,272,192 (10.75 MiB) |
| y2 | w2 (down proj) 输出 | [1, 512, 4096] | 2,097,152 | 4,194,304 (4.00 MiB) |
| **payload 总计** | | | **13,369,344** | **26,738,688 B = 25.50 MiB** |
| **每 round wire**（含 headers） | | | | **≈ 26.74 MB ≈ 213.94 Mbit** |

注：`silu(y1) * y3` 这个中间量 worker 算出但**不传**，coordinator 用收到的 y1/y3 自己重算后再校验 y2。

### 2.3 对比

| | per-round payload | per-round wire (含 header) | bits | 相对 attn-A |
|---|---|---|---|---|
| attn-llama (A: linear-only) | 16.00 MiB | 16.78 MB | 134.22 Mbit | 1.00× |
| **FFN** | **25.50 MiB** | **26.74 MB** | **213.94 Mbit** | **1.59×** |
| **attn-llama (B: with-scores)** | **32.00 MiB** | **33.55 MB** | **268.44 Mbit** | **2.00×** |

- **FFN 比 attn-A 多传 60%**：inter=11008 比 hidden=4096 大 2.7×，y1/y3 两个 [1,512,11008] tensor 撑大了总量
- **attn-B 比 attn-A 翻倍**：scores [1, 32, 512, 512] fp16 刚好 16 MiB，跟 Q/K/V/O 总和等大
- **attn-B 比 FFN 还多 25%**：在 GCP intra-zone 用 mode B 比跑 FFN 还更费带宽

---

## 3. GPU forward 时间（实测）

| Layer | GPU | GPU forward / round | 来源 |
|---|---|---|---|
| attn-llama | NVIDIA L4 | **2.1–3.0 ms** | 2026-05-14 实测，N=10 PIPELINE 各 round 平均 |
| FFN | (未确认型号) | **~2.18 ms** | [MULTI_MACHINE_FFN_REPORT.md](MULTI_MACHINE_FFN_REPORT.md) §4.1 |

理论 FLOP 量级（B=1, S=512）：
- attn-llama: 4 个 linear (q/k/v/o) + qk^T + probs@v ≈ **73 GFLOP**
- FFN: 3 个 linear (w1/w2/w3) ≈ **135 GFLOP**

L4 fp16 ~120 TFLOPS、A100 fp16 ~312 TFLOPS、H100 fp16 ~990 TFLOPS。理论：
- attn @ L4: 0.6 ms;  @ A100: 0.23 ms;  @ H100: 0.07 ms
- FFN  @ L4: 1.13 ms; @ A100: 0.43 ms;  @ H100: 0.14 ms

实测 ~2 ms 都比理论高一个数量级，原因是 batch=1 seq=512 太小，**kernel launch + RoPE + softmax 的固定开销占主导**，不是算力 bound。

---

## 4. 不同带宽下的理论传输时间

公式：`t_ms = bytes × 8 / bandwidth_bps`（不含 latency、不含 Python recv overhead）

### 4.1 各档带宽

| 带宽档位 | bandwidth | 来源 | attn-A 16.78 MB | FFN 26.74 MB | **attn-B 33.55 MB** |
|---|---|---|---|---|---|
| 1 GbE | 1 Gb/s | 老机房标杆 | 134.2 ms | 213.9 ms | **268.4 ms** |
| GCP-g2-default | 9.7 Gb/s | g2-standard-8 sustained egress（实测 cap） | 13.83 ms | 22.05 ms | **27.67 ms** |
| 10 GbE | 10 Gb/s | 标准 LAN | 13.42 ms | 21.39 ms | 26.84 ms |
| 25 GbE | 25 Gb/s | 中端数据中心 | 5.37 ms | 8.56 ms | 10.74 ms |
| GCP-c3-default | 32 Gb/s | c3-standard-44 NIC 物理上限 | 4.19 ms | 6.69 ms | 8.39 ms |
| 100 GbE | 100 Gb/s | 高端数据中心 | 1.34 ms | 2.14 ms | 2.68 ms |
| GCP-c3-tier1 | 100 Gb/s | c3 + Tier_1 networking | 1.34 ms | 2.14 ms | 2.68 ms |
| 200 GbE IB | 200 Gb/s | InfiniBand HDR | 0.67 ms | 1.07 ms | 1.34 ms |
| 400 GbE IB | 400 Gb/s | InfiniBand NDR | 0.34 ms | 0.53 ms | 0.67 ms |
| Local PCIe 4.0 x16 | 256 Gb/s | 同机箱 GPU↔CPU | 0.52 ms | 0.84 ms | 1.05 ms |

### 4.2 传输时间 / GPU 时间 比值

以 GPU forward = 2.1 ms (attn) / 2.18 ms (FFN) 为基准：

| 带宽档 | attn-A 比值 | FFN 比值 | **attn-B 比值** | 哪个 bound？ |
|---|---|---|---|---|
| 1 GbE | 63.9× | 98.1× | **127.8×** | Network 完全压制 |
| GCP-g2 (9.7G) | 6.59× | 10.11× | **13.18×** | Network 主导 |
| 10 GbE | 6.39× | 9.81× | 12.78× | Network 主导 |
| 25 GbE | 2.56× | 3.93× | 5.11× | Network 仍是瓶颈 |
| GCP-c3-default (32G) | 2.00× | 3.07× | **3.99×** | Network 仍主导 |
| 100 GbE / Tier_1 | 0.64× | 0.98× | **1.28×** | A 超临界，B 仍 network |
| 200 GbE IB | 0.32× | 0.49× | **0.64×** | GPU 主导 |
| 400 GbE IB | 0.16× | 0.24× | 0.32× | GPU 主导 |
| Local PCIe | 0.25× | 0.39× | 0.50× | GPU 主导 |

### 4.3 临界带宽

让 transfer time = GPU time 所需的带宽：

| Layer | payload (Mbit) | GPU (ms) | **临界带宽** |
|---|---|---|---|
| attn-llama (A: linear-only) | 134.2 | 2.1 | **64 Gb/s** |
| FFN | 213.9 | 2.18 | **98 Gb/s** |
| **attn-llama (B: with-scores)** | **268.4** | **2.1** | **128 Gb/s** |

→ **超过这个带宽，GPU 就是瓶颈了；低于这个，网络就是瓶颈**。

- attn-A 在 100 GbE 之上、FFN 在略超 100 GbE 时，GPU 开始顶住网络
- attn-B 临界翻倍到 128 Gb/s —— **100 GbE / Tier_1 还不够**，必须 200 GbE IB 才能让 GPU 主导
- GCP 默认 32 Gb/s 离 attn-A 临界差 2×，离 attn-B 临界差 4×

---

## 5. 实测 vs 理论：Python recv 的"协议税"

理论传输时间是物理链路极限。实际 Python `recv_exactly + frombuffer + copy` 单流远跑不到这个值。

### 5.1 单流实测

| Layer | 单 TCP 流实测 wire | 理论 (该带宽) | 效率 |
|---|---|---|---|
| FFN @ GCP-g2 (9.7 Gb/s) | **91.4 ms** | 22.05 ms | 24% |
| attn @ GCP-g2 (估算) | ~50 ms 量级 | 13.83 ms | ~28% |

数据来源：FFN report §4.1。单 TCP 流被 Python 反序列化卡死在 ~2.4 Gbit/s（远低于 NIC 9.6 Gbit/s）。

### 5.2 多流并发缓解

| 配置 | 聚合 wire BW | vs 链路上限 |
|---|---|---|
| FFN N=10 PIPELINE @ GCP-g2 | 9.60 Gb/s | **99% of g2 cap** |
| attn N=10 PIPELINE @ GCP-g2 | 6.26 Gb/s | 65% of g2 cap |

并发 N=10 把 FFN 推到 g2 egress cap (9.7 Gb/s)；attn 还差 35%（瓶颈在 coord verify CPU）。

### 5.3 实测 + 理论组合表（attn-llama）

| 带宽档 | 理论 wire (ms) | 单流实测 wire | N=10 聚合 wire | GPU (ms) | 谁主导？ |
|---|---|---|---|---|---|
| GCP-g2 (9.7G) | 13.8 | ~50 | 161 (per stream) | 2.1 | Network ×7 |
| GCP-c3 (32G) | 4.2 | — | — | 2.1 | Network ×2 |
| 100 GbE | 1.34 | — | — | 2.1 | **GPU** |

即使理论上 100 GbE 让 GPU 主导，实际 Python recv 单流的 ~28% 效率会让真实 wire 仍然 > GPU 时间。**协议层和 receiver 端的代码效率是隐藏瓶颈**。

---

## 6. 多 round vs GPU "纯计算"对比

把 GPU forward 当 baseline，看看协议在 N=10 PIPELINE 下相对 GPU 慢多少倍：

### 6.1 attn-llama (实测 2026-05-14, GCP-g2 worker)

| 方案 | per-round e2e | tok/s（B=1, S=512） | vs 纯 GPU 慢 |
|---|---|---|---|
| 纯 GPU (forward only) | 2.1 ms | ~244,000 | 1× |
| Multi-machine N=1 | ~210 ms | ~2,440 | **100×** |
| Multi-machine N=10 PIPELINE | 209 ms (per-stream) | 23,900 (聚合) | **10×** |

### 6.2 FFN (来自 FFN report)

| 方案 | tok/s | vs 纯 GPU 慢 |
|---|---|---|
| 纯 GPU baseline | ~236,000 | 1× |
| Multi-machine N=1 | 4,820 | 49× |
| Multi-machine N=10 PIPELINE | 22,030 | **~10.7×** |

→ 两个 layer 在 N=10 PIPELINE + GCP intra-zone 下都是 **~10× 慢于纯 GPU**，差距完全来自 wire 传输 + Python overhead，不是 GPU。

### 6.3 如果带宽升到 Tier_1 / 多 worker VM

按理论传输时间 + ~28% Python 效率推算：

| 配置 | 单流理论 wire | 单流实际 wire (×3.5) | tok/s 估算 (per stream) |
|---|---|---|---|
| GCP-g2 (9.7 Gb/s) | 13.8 ms | ~50 ms | 10,200 |
| GCP-c3 (32 Gb/s) | 4.2 ms | ~15 ms | 33,000 |
| 100 GbE / Tier_1 | 1.3 ms | ~5 ms | 100,000 |
| Local PCIe | 0.5 ms | ~2 ms (≈ GPU) | 250,000 (≈纯 GPU) |

→ 升带宽线性提升直到撞 GPU + Python 协议层；要继续提升需要**优化 receiver 端代码**（C extension recv、或者 verify 移到 GPU）。

### 6.4 attn 模式 B (with-scores) 的成本/收益

把 qk^T 也搬上 wire 的代价 vs 价值（loopback 实测，GPU=CPU 同机）：

| 指标 | 模式 A (linear-only) | **模式 B (with-scores)** | Δ |
|---|---|---|---|
| 每 round wire | 16.78 MB | **33.55 MB** | **+100%** |
| @ GCP-g2 (9.7 Gb/s) 理论 wire | 13.83 ms | 27.67 ms | +13.84 ms |
| @ Tier_1 (100 Gb/s) 理论 wire | 1.34 ms | 2.68 ms | +1.34 ms |
| coord verify CPU（recompute） | ~35 ms attn-recompute（隐含在 o-verify 里） | + ~24 ms scores-recompute（实测） | **+24 ms** |
| 端到端（loopback hidden=4096） | 138.8 ms | 235.0 ms | **+69%** |

**结论：mode B 在当前实现下纯亏**：
- 多 ~14 ms wire（GCP-g2）+ ~24 ms CPU（coord 重算 scores 来对比）
- 没有省掉任何东西（o-verify 的 attn-recompute 仍要跑，因为我们没有信任收到的 scores 直接用）

**为什么仍然实现这个模式**：
1. 量化"**也传 attention 内部需要多大带宽**"：临界带宽从 64 → 128 Gb/s（必须 200 GbE IB 才能让 GPU 主导）
2. 为后续 **Freivalds 验证** 打基础：未来如果 coord 用 r∈R^[S,k] 做 `scores @ r ≟ q @ (k^T @ r)` 验证（O(S·k) 而非 O(S²·head_dim)），就能跳过 ~35 ms 的 attn-recompute，net gain 才是正的

→ Mode B 当前主要用于**实验/校准**，不建议生产用。生产仍用默认的 mode A。

---

## 7. 总结

### 7.1 数据量层面

- **attn-llama mode A (默认)**: 16.78 MB/round (Q+K+V+O 各 4 MB)
- **FFN**: 26.74 MB/round (y1+y3 各 10.75 MB + y2 4 MB)
- **attn-llama mode B (with-scores)**: 33.55 MB/round (mode A + 16 MB scores)
- 排序：attn-A (1.0×) < FFN (1.6×) < attn-B (2.0×)

### 7.2 GPU 计算层面

- 都是 **~2 ms / round** 量级（kernel launch overhead 主导）
- 即使升级到 H100，也只能压到 0.1-0.2 ms 量级
- **GPU 永远不是这套协议的瓶颈**

### 7.3 网络/传输层面

- **临界带宽**：attn-A 64 Gb/s、FFN 98 Gb/s、**attn-B 128 Gb/s** — 超过此值 GPU 才会成为瓶颈
- **GCP 默认 (g2 9.7 Gb/s)**: 比 GPU 慢 7-13×（网络完全主导）
- **GCP c3 默认 (32 Gb/s)**: 比 GPU 慢 2-4×
- **GCP c3 + Tier_1 (100 Gb/s)**: attn-A 接近平衡、FFN 刚好平、**attn-B 仍 1.3× network-bound**
- **本地 PCIe (256 Gb/s)**: 全部 GPU 主导

### 7.4 实际优化方向

按收益从大到小：

1. **加 worker VM 数量**（横向扩展 g2 egress cap）—— 单台 9.7 Gb/s 是硬墙，加机器线性扩
2. **coord 升 Tier_1**（免费，c3 支持）—— NIC 32 → 100 Gb/s
3. **降 wire 字节**（fp8/int8 / 只传摘要）—— 直接把 16/26 MB 砍下来
4. **C extension recv path**（替换 Python `recv_exactly + copy`）—— 单流 effective 从 2.4 Gb/s 拉到 5+ Gb/s
5. **coord verify 移到 GPU**（如果 coord 有 GPU）—— SLALOM matmul GPU 化

GPU 端、协议帧头、SLALOM 算法本身**都不值得优化**，全在网络这一侧。
