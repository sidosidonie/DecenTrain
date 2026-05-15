


# FFN Example


| 项 | 值 |
|---|---|
| FFN | SwiGLU MLP,`hidden=4096`,`inter=11008` |
| 输入 | `batch=1`,`seq=512`(每轮 512 tokens) |
| Wire dtype | `fp16` |
| SLALOM `k` | 10 |
| Coordinator 机型 | 44 vCPU,171 GB RAM,无 GPU, 32Gbps |
| Worker 机型 |  g2-standard-4 (4 vCPUs, 16 GB Memory, L4 24GB), 10Gbps |
| 每次 run | 100 rounds(10 warmup,90 measured) |
| 多流测试 | N=5(端口 9100–9104)、N=10(端口 9100–9109) |


Shape:

```
y1 = w1(x)              # gate 投影
y3 = w3(x)              # up 投影
y2 = w2(silu(y1) * y3)  # down 投影
```

形状(本测试):`hidden=4096, inter=11008, batch=1, seq=512`,wire dtype `fp16`。


| Tensor | 来自的 matmul | 形状 | 字节(fp16) |
|---|---|---|---|
| `y1 (w1)` | `y1 = w1(x)` | `[1, 512, 11008]` | 11.27 MB |
| `y3 (w3)` | `y3 = w3(x)` | `[1, 512, 11008]` | 11.27 MB |
| `y2 (w2)` | `y2 = w2(silu(y1) * y3)` | `[1, 512, 4096]` | 4.19 MB |
| **合计** | | | **26.74 MB** |


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


## Summary
- worker受到
- 通过在一个GPU机器上增加instance数量，提升GPU测的带宽，GPU任务的latency会收到GPU的制约，


## Google cloud 上机器的带宽

  https://cloud.google.com/products/compute/pricing/general-purpose
- 和机型有关
- CPU端可以打开Tier1，网络最高可达200Gbps($730.00 / 1 month)
  https://cloud.google.com/blog/products/compute/increasing-bandwidth-to-compute-engine-vms-with-tier_1-networking
- GPU机器不支持Tier1，和GPU本身有关系。

| 机型           | GPU   | vCPU | 默认带宽上限 |
|----------------|-------|------|--------------|
| g2-standard-4  | 1× L4 | 4    | 10 Gbps      |
| g2-standard-8  | 1× L4 | 8    | 16 Gbps      |
| g2-standard-12 | 1× L4 | 12   | 16 Gbps      |
| g2-standard-16 | 1× L4 | 16   | 32 Gbps      |
| g2-standard-24 | 2× L4 | 24   | 32 Gbps      |
| g2-standard-32 | 1× L4 | 32   | 32 Gbps      |
| g2-standard-48 | 4× L4 | 48   | 50 Gbps      |
| g2-standard-96 | 8× L4 | 96   | 100 Gbps     |

PCIE带宽

| 代次 | x16 单向 | x16 双向 (聚合) | 典型 NVIDIA GPU                             |
|------|----------|-----------------|---------------------------------------------|
| Gen1 | 4 GB/s   | 8 GB/s          | —                                           |
| Gen2 | 8 GB/s   | 16 GB/s         | —                                           |
| Gen3 | 16 GB/s  | 32 GB/s         | P100, V100, T4, RTX 20                      |
| Gen4 | 32 GB/s  | 64 GB/s         | A100, A30/40, L4, L40, RTX 30/40, H100 PCIe |
| Gen5 | 64 GB/s  | 128 GB/s        | H100 SXM5, H200, B100/B200, RTX 50, MI300X  |
| Gen6 | 121 GB/s | 256 GB/s        |                                             |


## 各档带宽

| 带宽档位           | bandwidth | 来源                                       | attn-A 16.78 MB | FFN 26.74 MB | **attn-B 33.55 MB** |
|--------------------|-----------|--------------------------------------------|-----------------|--------------|---------------------|
| 1 GbE              | 1 Gb/s    | 老机房标杆                                 | 134.2 ms        | 213.9 ms     | **268.4 ms**        |
| GCP-g2-default     | 9.7 Gb/s  | g2-standard-8 sustained egress（实测 cap） | 13.83 ms        | 22.05 ms     | **27.67 ms**        |
| 10 GbE             | 10 Gb/s   | 标准 LAN                                   | 13.42 ms        | 21.39 ms     | 26.84 ms            |
| 25 GbE             | 25 Gb/s   | 中端数据中心                               | 5.37 ms         | 8.56 ms      | 10.74 ms            |
| GCP-c3-default     | 32 Gb/s   | c3-standard-44 NIC 物理上限                | 4.19 ms         | 6.69 ms      | 8.39 ms             |
| 100 GbE            | 100 Gb/s  | 高端数据中心                               | 1.34 ms         | 2.14 ms      | 2.68 ms             |
| GCP-c3-tier1       | 100 Gb/s  | c3 + Tier_1 networking                     | 1.34 ms         | 2.14 ms      | 2.68 ms             |
| 200 GbE IB         | 200 Gb/s  | InfiniBand HDR                             | 0.67 ms         | 1.07 ms      | 1.34 ms             |
| Local PCIe 4.0 x16 | 256 Gb/s  | 同机箱 GPU↔CPU                             | 0.52 ms         | 0.84 ms      | 1.05 ms             |

## 传输时间 / GPU 时间 比值

以 GPU forward = 2.1 ms (attn) / 2.18 ms (FFN) 为基准：

| 带宽档               | attn-A 比值 | FFN 比值 | **attn-B 比值** | 哪个 bound？           |
|----------------------|-------------|----------|-----------------|------------------------|
| 1 GbE                | 63.9×       | 98.1×    | **127.8×**      | Network 完全压制       |
| GCP-g2 (9.7G)        | 6.59×       | 10.11×   | **13.18×**      | Network 主导           |
| 10 GbE               | 6.39×       | 9.81×    | 12.78×          | Network 主导           |
| 25 GbE               | 2.56×       | 3.93×    | 5.11×           | Network 仍是瓶颈       |
| GCP-c3-default (32G) | 2.00×       | 3.07×    | **3.99×**       | Network 仍主导         |
| 100 GbE / Tier_1     | 0.64×       | 0.98×    | **1.28×**       | A 超临界，B 仍 network |
| 200 GbE IB           | 0.32×       | 0.49×    | **0.64×**       | GPU 主导               |
| Local PCIe           | 0.25×       | 0.39×    | 0.50×           | GPU 主导               |


