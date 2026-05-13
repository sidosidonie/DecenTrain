# Multi-Machine FFN 性能测试报告

> SwiGLU FFN 单层 · CPU 协调端 ↔ GPU 工作端 · 输出仅传 + SLALOM 校验
> 基于 `examples/multi_machine_ffn.py` 的实测

---

## 1. 概述

本文档记录在两台 GCP VM 之间运行 `examples/multi_machine_ffn.py`(单层 SwiGLU MLP,带 SLALOM 概率校验的"输出仅传"协议)的实测性能与瓶颈分析。

- **测试日期**:2026-05-13
- **测试目标**:量化一个被密码学校验的 GPU FFN forward 在多机部署下的端到端开销,并定位瓶颈
- **核心结论**:正确性 100%,GPU 计算只占总时间的 ~2%,瓶颈在 wire 传输;通过多路并发 (N=10) + pipeline overlap,聚合吞吐做到纯 GPU 的 **~9.3%**(约 10.7× 慢),且接近 10 GbE 网卡饱和

---

## 2. 架构与原理

### 2.1 两个角色

| 角色 | 机器 | 职责 | 信任级别 |
|---|---|---|---|
| **Coordinator(协调端)** | `10.128.0.2`(CPU,44 vCPU / 171 GB) | 拥有 SLALOM 投影向量 `s` / `s_tilde`;从共享 seed 重生输入 `x`;接收 worker 返回的 matmul 输出,逐个做概率校验 | 受信(可被部署在 TEE 内) |
| **Worker(工作端)** | `10.128.0.3`(GPU) | 加载 SwiGLU MLP 权重,执行 forward,把三个 linear 的输出回传 | 不受信(可能是恶意 GPU) |

### 2.2 SwiGLU MLP

测试的是 LLaMA 风格的单层 FFN(无 bias),三次 matmul:

```
y1 = w1(x)              # gate 投影
y3 = w3(x)              # up 投影
y2 = w2(silu(y1) * y3)  # down 投影
```

形状(本测试):`hidden=4096, inter=11008, batch=1, seq=512`,wire dtype `fp16`。

### 2.3 Wire 协议(输出仅传)

每轮 forward 在两台机器之间走 TCP,**只传四类消息体**:

| 消息 | 方向 | 内容 | 大小 |
|---|---|---|---|
| `MSG_LOAD_REQ` | C → W | `(hidden, inter, weight_seed, dtype_id)` | 16 B |
| `MSG_LOAD_ACK` | W → C | status byte | 1 B |
| `MSG_FORWARD_REQ` | C → W | `(request_id, input_seed, batch, seq)` | 20 B |
| **`MSG_ACTIVATION`** ×3 | W → C | header + shape + **tensor bytes**(y1 / y3 / y2) | **26.74 MB / round** |
| `MSG_FORWARD_DONE` | W → C | `(request_id, gpu_forward_t_ms)` | 16 B |

每轮 worker 回传的 tensor:

| Tensor | 来自的 matmul | 形状 | 字节(fp16) |
|---|---|---|---|
| `y1 (w1)` | `y1 = w1(x)` | `[1, 512, 11008]` | 11.27 MB |
| `y3 (w3)` | `y3 = w3(x)` | `[1, 512, 11008]` | 11.27 MB |
| `y2 (w2)` | `y2 = w2(silu(y1) * y3)` | `[1, 512, 4096]` | 4.19 MB |
| **合计** | | | **26.74 MB** |

**worker 不传**的内容:
- 输入 `x` —— worker 端从 `input_seed` 自己用 `torch.randn` 重生
- 权重 `w1/w2/w3` —— 两端各自用 `weight_seed` 调 `make_weights` 同步生成
- 中间量 `silu(y1) * y3` —— GPU 算出但不上链;coordinator 用收到的 y1/y3 自己重算
- SLALOM 投影向量 `s`/`s_tilde` —— 只在 coordinator 一侧存在

校验:**预测字节 = 实测字节 = 26.74 MB**(每次跑都吻合),证明了 wire 上确实**只有**这三个矩阵输出。

### 2.4 SLALOM 校验思路(`k=10`)

针对每个 linear `y = W·x`:coordinator 持有一组随机投影 `s`(维度 `out`) 和它们的"预投影" `s_tilde = W^T·s`(维度 `in`);收到 worker 算出的 `y` 之后:

- 计算 `<y, s>` 和 `<x, s_tilde>`(各是 k=10 维的小向量)
- 二者应严格相等(`<W·x, s> = <x, W^T·s>`),取均方误差(mse)与阈值比较

阈值随 wire dtype/dims 自适应:`fp16` 下约为 `inter * 2e-6 = 0.022`(本测试)。实测 `mse_p95` ≈ `2.4e-3 / 2.4e-3 / 5.0e-3`,**比阈值低一个数量级**,且每次 run 完全相同(seed 决定)。

整层 FFN 用 3 次 SLALOM 校验(对应 3 个 linear),时间复杂度从 `O(B·S·H·I)` 降到 `O(B·S·(H+I)·k)`,在本配置下校验只花 14–15 ms。

---

## 3. 测试配置

| 项 | 值 |
|---|---|
| FFN | SwiGLU MLP,`hidden=4096`,`inter=11008` |
| 输入 | `batch=1`,`seq=512`(每轮 512 tokens) |
| Wire dtype | `fp16` |
| SLALOM `k` | 10 |
| Coordinator 机型 | 44 vCPU,171 GB RAM,无 GPU |
| Worker 机型 | 有 GPU(具体型号未确认) |
| 链路 | GCP 同 zone 内网,名义 10 GbE |
| 每次 run | 100 rounds(10 warmup,90 measured) |
| 多流测试 | N=5(端口 9100–9104)、N=10(端口 9100–9109) |

---

## 4. 单流(N=1)实测

5 次连续 run 取均值。所有 run 均 **90/90 通过校验**。

### 4.1 端到端 & 阶段时间

| 指标 | 均值 | 范围 | 占 E2E |
|---|---|---|---|
| 端到端 / round | **106.4 ms** | 102.4 – 110.9 | 100% |
| **吞吐** | **9.41 round/s ≈ 4 820 tokens/s** | 9.01 – 9.77 | — |
| GPU forward(3 个 matmul) | **2.18 ms** | 2.17 – 2.18 | **~2.0%** |
| Wire recv(26.7 MB) | **91.4 ms** | 87.2 – 96.1 | **~83.7%** |
| CPU SLALOM 校验 | **14.9 ms** | 14.8 – 15.2 | **~14.2%** |
| Wire 有效带宽 | **~2.4 Gbit/s**(~297 MB/s) | 283.6 – 310.4 MB/s | — |
| GPU 算力 | ~63.7 TFLOPS(fp16) | 63.6 – 63.8 | — |
| 三相加和(`sum_pct`) | ~99.9%(sequential ≈100% ✓) | — | — |

### 4.2 关键观察

1. **GPU 几乎闲着**(2.18 ms 占 2%)。138 GFLOP 的活在 2.17 ms 跑完,有效 63.7 TFLOPS,但因为 batch×seq 太小,小 matmul 是 latency-bound,远没吃满卡——不重要,反正它不是瓶颈。
2. **Wire 传输吃掉 84%**,有效带宽只有 ~2.4 Gbit/s ≈ 名义 10 GbE 的 23%。这是单条 TCP 流 + Python 反序列化(`recv_exactly` → `np.frombuffer` → `reshape` → `copy` → `torch.from_numpy` → `.to(fp32)`)的吞吐天花板,**不是物理链路**(下文 N=10 实测会证实)。
3. **校验稳定 ~15 ms**(占 14%);3 个 linear 中 `w2` 维度最小(`inter→hidden`)所以最快(~4.8 ms),`w1/w3` 各 ~7.1 ms。
4. **正确性绝对稳定**:5 次 × 90 rounds 全过,mse 字节级一致(seed 完全决定结果)。

### 4.3 Wire 估算(对比理想)

每轮 26.74 MB(= 214 Mbit)在不同名义带宽下的理论传输时间:

| 名义链路 | 理论时间 | 实测(on-wire) | 效率 |
|---|---|---|---|
| 1 Gbit/s | 213.9 ms | — | — |
| **10 Gbit/s** | 21.4 ms | **92.6 ms** | **23.1%** |
| 25 Gbit/s | 8.6 ms | — | 9.2% |

实测 on-wire 92.6 ms 对应 **~2.31 Gbit/s 有效带宽**;对 10 GbE 来说只有 23% 利用率——这是单流的"Python 协议税"。

---

## 5. 多流扩展(N=5,N=10)

在 worker 机上跑 N 个 worker 进程(各占一个端口),coordinator 端用 `run_cpu_coordinator_fanout.sh` 并发起 N 个 coordinator 进程,各连一个端口,各自完成 100 rounds。

### 5.1 N=5(sequential)

| 指标 | 值 |
|---|---|
| Aggregate round/s | **25.83** |
| 加速比 vs N=1 | **2.74×**(理想 5×) |
| 扩展效率 | **55%** |
| 每流 round/s | 5.03 – 5.32(均 5.16) |
| 每流 wire ms | 173 – 183(均 177) |
| 每流 verify ms | 15.1 – 17.3(均 16.4) |
| 每流有效带宽 | 1.18 – 1.26 Gbit/s |
| **Aggregate 有效带宽** | **6.12 Gbit/s**(61% of 10 GbE) |
| 通过率 | **450/450** ✓ |

### 5.2 N=10(sequential 与 pipeline 对比)

| 指标 | N=10 sequential | **N=10 PIPELINE=1** | Δ |
|---|---|---|---|
| Aggregate round/s | 37.89 | **43.02** | **+13.5%** |
| Aggregate tokens/s | ~19 400 | **~22 030** | +13.6% |
| 加速比 vs N=1 | 4.03× | **4.57×** | — |
| 扩展效率(vs 10× 理想) | 40% | **46%** | — |
| 每流 round/s | 3.79 | 4.30 | +13.5% |
| 每流 p50 端到端 | 264 ms | **233 ms** | −12% |
| 每流 wire ms | 234 | 226 | −3% |
| 每流 verify ms(墙上) | 30.7 | 96.2 *(overlapping)* | — |
| `sum_pct`(总占比) | ~100%(sequential) | **137%**(overlap working) | — |
| Aggregate 有效带宽 | 9.32 Gbit/s | **9.60 Gbit/s** | +3% |
| 链路利用(vs 10 GbE) | 93% | **96%** | — |
| 通过率 | **900/900** | **900/900** | ✓ |

> Pipeline 模式下 verify 看上去从 31 → 96 ms,是因为它在 pipeline 中是**墙上时间**(从第一个 verify submit 到最后一个 result),完全 overlap 在 wire 窗口里;`sum_pct = 137%` 证明 wire 96.5% + verify 39% 在同一时刻并行,验证 overlap 真的在生效。"额外耗时"实际接近 0。

### 5.3 N=1 → N=5 → N=10 扩展曲线

| N | Aggregate round/s | Aggregate tok/s | Aggregate eff bw | 链路利用 | 扩展效率 |
|---|---|---|---|---|---|
| 1 | 9.41 | ~4 820 | 2.4 Gbit/s | 23% | 100% |
| 5 | 25.83 | ~13 200 | 6.12 Gbit/s | 61% | 55% |
| 10 (seq) | 37.89 | ~19 400 | 9.32 Gbit/s | 93% | 40% |
| **10 (pipe)** | **43.02** | **~22 030** | **9.60 Gbit/s** | **96%** | **46%** |

**结论**:**聚合带宽在 N=10 时已逼近 10 GbE 实际上限**,继续加流收益将快速衰减(网卡先碰到顶,然后 coordinator 端 80 OMP 线程在 44 vCPU 上的争用也会让 verify 时间继续涨)。

---

## 6. 与"纯 GPU"对比

把测出来的 GPU forward 时间(2.17 ms / round = 1 layer × 512 tokens)作为纯 GPU 基线:

| 方案 | tokens/s | 比纯 GPU 慢 |
|---|---|---|
| **纯 GPU(基线)** | **~236 000** | **1×** |
| 多机 N=1 sequential | 4 820 | **48.9×** |
| 多机 N=5 sequential(聚合) | 13 200 | 17.9× |
| 多机 N=10 sequential(聚合) | 19 400 | 12.2× |
| **多机 N=10 PIPELINE=1(聚合)** | **22 030** | **~10.7×** |

**解读**:这套"输出仅传 + SLALOM 校验"协议的代价,本质是用 **~10.7× 吞吐下降 / 50× 单流延迟下降**,换来"GPU 算的输出可被 CPU(TEE 一侧)以可忽略误差概率被密码学校验"这个安全属性。GPU 本身从来不是瓶颈(全程只占 2% 时间)。

注:
- 这是 **1×512 的小 batch** 工况,纯 GPU 也才 ~20% 显卡利用率(63 TFLOPS / A100 fp16 峰值 312 TFLOPS),所以"纯 GPU 基线"已偏低估;真要打满 GPU,差距还会拉大。
- 这是**单层 FFN**。整模型有几十层,绝对开销线性放大,**相对比例(~10×)不变**(只要协议不变)。

---

## 7. 瓶颈分析

按"在何种工况下变成瓶颈"排序:

| # | 资源 | 单流瓶颈点 | 多流瓶颈点 | 说明 |
|---|---|---|---|---|
| 1 | **单条 TCP 流 + Python 反序列化** | **是**(占 84%,只跑出 2.4 Gbit/s) | 已被多流解开 | 单 `recv` 循环 + `np.frombuffer` + `to(fp32)` 单线程喂不动 10 GbE |
| 2 | **物理 NIC / GCP egress 配额** | 远未饱和 | **是**(N=10 时 9.6/10 Gbit/s,96% 利用) | 真实链路上限,多流并发已逼近 |
| 3 | CPU(coordinator 端 SLALOM 校验) | 不是(14% 占比) | **开始出现**(N=10 时 verify 15→31 ms) | 10 进程 × 8 OMP 线程 = 80 线程跑在 44 vCPU,争用产生 |
| 4 | GPU forward | **从不**(2% 占比,2.17 ms) | 不是 | matmul launch-bound,batch 大时也只线性放大 |
| 5 | 协议固定开销(framing / serial.) | 小(每 message 8B header,占 < 0.1%) | 小 | 已经很紧 |

### 7.1 单流为什么只有 23% 链路利用率?

26.7 MB 拆成 3 个 tensor,每个都要走:`recv_exactly`(逐 chunk read)→ `np.frombuffer`(create view)→ `arr.copy()`(让 buffer 可写)→ `torch.from_numpy(...).to(fp16)` → `tensor.to(float32)`。这条链路是**纯单线程 Python + 一次内存拷贝**,在小消息上每秒能搬约 300 MB,折合 2.4 Gbit/s——和实测完全吻合。

### 7.2 N=10 为什么扩展效率只有 46%?

理想 10×,实际 4.57×。多出来的 ~5× 损失来自:
- **网卡饱和**(主因):聚合 9.6/10 Gbit/s,继续加流也压不出更多带宽 → 每流被迫排队,wire 时间从 91 → 226 ms。
- **CPU 争用**(次因):80 OMP 线程在 44 vCPU,SLALOM 矩阵乘抢核,verify 时间从 15 → 31 ms。Pipeline overlap 缓解了这部分(让 verify 藏在 wire 后面)但没消除争用本身。

### 7.3 GPU 为什么"永远不是瓶颈"?

- 实测 GPU forward 2.17 ms(63.7 TFLOPS),wire 92 ms。比例 1 : 42。
- 即使把 batch×seq 翻 10 倍(到 5120 tokens):
  - GPU 时间 ≈ 21.7 ms(线性增长,但仍 launch-bound)
  - Wire 字节翻 10 倍 → 267 MB,以 ~2.4 Gbit/s 算需 ~890 ms
  - 比例变成 1 : 41 —— **几乎不变**
- 协议设计上"完整输出回传"决定了 wire 永远是 GPU 的几十倍。

---

## 8. 优化方向

按预期收益从大到小:

### 已尝试

1. **多流并发(N=10)**:聚合吞吐 **+4.03×**(单流 → N=10 sequential),把链路从 23% 推到 93% 利用。
2. **Pipeline overlap**:在 N=10 上额外 **+13.5%** 聚合吞吐,通过把 verify 藏在 receive 窗口里把端到端延迟从 264 → 233 ms。

### 未尝试,有潜力

3. **改 wire dtype 为 bf16/fp8/int8** 或**只传摘要**(SLALOM 实际只需 k 个 dot product):传输量直接减半到一个数量级,直接攻击 wire 这个 84% 的瓶颈。是最大杠杆。
4. **多 NIC / 升级机型**:在 GCP 上 N1/N2 可获更高 egress 上限;真要 25 Gbps + 多队列,链路上限可推到 ~20 Gbit/s 量级。
5. **C/Cython recv path**:把 `recv_exactly` + `frombuffer` + `copy` 替换为零拷贝/共享内存版本,单流上限可能从 2.4 Gbit/s 提到 5+ Gbit/s。代价是工程复杂度。
6. **大 batch**:相对比例不变,但**每 round 固定开销(framing 等)被摊薄**,且 GPU 利用率上升,从经济角度划算。
7. **Coordinator 端限制 OMP/torch 线程数 + CPU pin**:在高 N 下减少调度开销,可能让 verify 时间不涨这么快。

### 已饱和,不要再尝试

- **继续加 N**:N=15、N=20 在当前 10 GbE 下大概率聚合吞吐不会再涨,verify 时间会继续恶化。
- **worker 侧 PIPELINE=1**:worker 上的 send-with-compute overlap,理论收益 < 2 ms(GPU forward 才 2.17 ms),不值得专门重启。

---

## 9. 复现命令

```bash
# 一、单流 baseline(coordinator 这边):
WORKER_HOST=10.128.0.3 LINK_GBPS=10 examples/run_cpu_coordinator.sh

# 二、N=10 fan-out(coordinator 这边):
WORKER_HOST=10.128.0.3 N=10 PORT_BASE=9100 ROUNDS=100 WARMUP=10 \
    LINK_GBPS=10 PIPELINE=1 examples/run_cpu_coordinator_fanout.sh

# 三、N=10 workers(worker 机上):
N=10 DEVICE=cuda examples/run_workers_fanout.sh
#   或 DEVICE=cpu / DEVICE=cuda:0,cuda:1 / 等
```

输出:
- 每流的 `coord_i.json`(per-round 数据 + summary)和 `coord_i.log`(完整 printed summary)
- 汇总表打印到 stdout

---

## 10. 结论

1. **协议正确性绝对稳定**:N=1/5/10 共 11 次 run,**1 800 / 1 800 rounds 全过**;mse 字节级一致(seed 决定)。
2. **GPU 不是瓶颈**(全程 2% 时间)——这是"输出仅传"协议的本质特征,无论硬件多强,GPU 都在等链路。
3. **单流被 Python 反序列化卡死在 ~2.4 Gbit/s**,只占 10 GbE 的 23%——这不是"网络差",是单条 TCP 流上 Python `recv` + `frombuffer` 的吞吐天花板。
4. **多流(N=10)+ pipeline 把链路推到 ~96% 名义利用**(9.6/10 Gbit/s),聚合 4.57× 加速,这是当前协议/链路下接近最优的状态。
5. **整体相对纯 GPU 慢 ~10.7×**(吞吐)/ ~50×(单流延迟),这是用于换取"GPU 输出被密码学校验"这一安全属性的代价。
6. **下一步最大杠杆**:减少 wire 字节数(更小 dtype 或只传摘要),其次升级链路带宽。GPU、校验、协议帧头都不是值得优化的方向。

---

## 附录:测试用文件清单

| 文件 | 作用 |
|---|---|
| `examples/multi_machine_ffn.py` | 主程序,coordinator/worker 共用 |
| `examples/run_cpu_coordinator.sh` | 单流 coordinator 启动 |
| `examples/run_gpu_worker.sh` | 单 worker 启动 |
| `examples/run_cpu_coordinator_fanout.sh` | N 个 coordinator 并发 + 汇总 |
| `examples/run_workers_fanout.sh` | N 个 worker 一键起 |
| `docs/MULTI_MACHINE.md` | 协议设计文档(原始) |
| **本文档** | **实测性能报告** |
