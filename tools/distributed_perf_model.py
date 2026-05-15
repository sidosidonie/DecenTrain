"""
Distributed Verification Performance Model

Analytical model for predicting latency and throughput of the distributed
verification architecture (1 CPU TEE + N GPU workers).

Calibrated against real benchmark data (Qwen3.5-9B, RTX 5090, 18 cores):
  Real: 137ms origin forward, 1950ms verified forward (14.3x)
  Per-check: linear_freivalds avg 56.14ms, matmul_freivalds avg 13.52ms
  D2H: linear avg 0.64ms, matmul avg 2.69ms
  SLALOM estimated: ~0.3ms per linear check (190x faster)

Two verification modes:
  - Standard Freivalds: O(n^2*k), reads full weight matrix (memory-BW bound)
  - SLALOM preprocessed: O(n*k), reads only small vectors (cache-friendly)

Usage:
  python tools/distributed_perf_model.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


# ── Hardware Specifications ─────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_verified_attn_layers: int
    num_verified_mlp_layers: int
    param_bytes: float  # total model size in bytes (bf16)

    @property
    def total_verified_layers(self) -> int:
        return max(self.num_verified_attn_layers, self.num_verified_mlp_layers)

    @property
    def linear_checks_per_fwd(self) -> int:
        """Number of linear layer verifications per forward pass."""
        # Attention: q, k, v, o = 4 per attn layer
        # MLP: gate, up, down = 3 per MLP layer
        return self.num_verified_attn_layers * 4 + self.num_verified_mlp_layers * 3

    @property
    def matmul_checks_per_fwd(self) -> int:
        """Number of dynamic matmul verifications per forward pass (Q@K^T, P@V)."""
        return self.num_verified_attn_layers * 2

    @property
    def elementwise_checks_per_fwd(self) -> int:
        """Softmax recompute checks."""
        return self.num_verified_attn_layers * 1


MODELS = {
    "Qwen2.5-0.5B": ModelSpec(
        name="Qwen2.5-0.5B", hidden_size=896, intermediate_size=4864,
        num_layers=24, num_attention_heads=14, num_kv_heads=2, head_dim=64,
        num_verified_attn_layers=24, num_verified_mlp_layers=24, param_bytes=1e9,
    ),
    "Llama-3.2-1B": ModelSpec(
        name="Llama-3.2-1B", hidden_size=2048, intermediate_size=8192,
        num_layers=16, num_attention_heads=32, num_kv_heads=8, head_dim=64,
        num_verified_attn_layers=16, num_verified_mlp_layers=16, param_bytes=2e9,
    ),
    "Qwen3.5-9B": ModelSpec(
        name="Qwen3.5-9B", hidden_size=4096, intermediate_size=12288,
        num_layers=32, num_attention_heads=32, num_kv_heads=8, head_dim=128,
        num_verified_attn_layers=8, num_verified_mlp_layers=32, param_bytes=18e9,
    ),
    "Llama-3-70B": ModelSpec(
        name="Llama-3-70B", hidden_size=8192, intermediate_size=28672,
        num_layers=80, num_attention_heads=64, num_kv_heads=8, head_dim=128,
        num_verified_attn_layers=80, num_verified_mlp_layers=80, param_bytes=140e9,
    ),
    "Qwen2.5-72B": ModelSpec(
        name="Qwen2.5-72B", hidden_size=8192, intermediate_size=29568,
        num_layers=80, num_attention_heads=64, num_kv_heads=8, head_dim=128,
        num_verified_attn_layers=80, num_verified_mlp_layers=80, param_bytes=144e9,
    ),
}


@dataclass
class GPUSpec:
    name: str
    bf16_tflops: float
    memory_gb: float
    memory_bw_gbps: float  # HBM bandwidth GB/s

GPUS = {
    "RTX-4090": GPUSpec("RTX 4090", bf16_tflops=165, memory_gb=24, memory_bw_gbps=1008),
    "RTX-5090": GPUSpec("RTX 5090", bf16_tflops=419, memory_gb=32, memory_bw_gbps=1792),
    "A100-80G": GPUSpec("A100 80GB", bf16_tflops=312, memory_gb=80, memory_bw_gbps=2039),
    "H100-SXM": GPUSpec("H100 SXM", bf16_tflops=990, memory_gb=80, memory_bw_gbps=3350),
    "H200": GPUSpec("H200", bf16_tflops=990, memory_gb=141, memory_bw_gbps=4800),
    "B200": GPUSpec("B200", bf16_tflops=2250, memory_gb=192, memory_bw_gbps=8000),
}


@dataclass
class CPUSpec:
    name: str
    cores: int
    fp32_gflops: float  # total FP32 GFLOPS
    memory_bw_gbps: float  # DRAM bandwidth GB/s

CPUS = {
    "bench-18core": CPUSpec("Bench 18-core", cores=18, fp32_gflops=900, memory_bw_gbps=90),
    "Xeon-8480+": CPUSpec("Xeon 8480+ 56C", cores=56, fp32_gflops=2150, memory_bw_gbps=307),
    "Xeon-w9-3595X": CPUSpec("Xeon w9-3595X 60C", cores=60, fp32_gflops=2600, memory_bw_gbps=307),
    "EPYC-9654": CPUSpec("EPYC 9654 96C", cores=96, fp32_gflops=3072, memory_bw_gbps=461),
    "EPYC-9754": CPUSpec("EPYC 9754 128C", cores=128, fp32_gflops=3200, memory_bw_gbps=461),
    "2xEPYC-9754": CPUSpec("2xEPYC 9754 256C", cores=256, fp32_gflops=6400, memory_bw_gbps=922),
    "i9-14900K": CPUSpec("i9-14900K 24C", cores=24, fp32_gflops=900, memory_bw_gbps=90),
}


@dataclass
class NetworkSpec:
    name: str
    bandwidth_gbps: float
    latency_us: float

NETWORKS = {
    "1GbE": NetworkSpec("1 GbE", bandwidth_gbps=1, latency_us=500),
    "10GbE": NetworkSpec("10 GbE", bandwidth_gbps=10, latency_us=100),
    "25GbE": NetworkSpec("25 GbE", bandwidth_gbps=25, latency_us=50),
    "100GbE": NetworkSpec("100 GbE", bandwidth_gbps=100, latency_us=10),
    "200GbE-IB": NetworkSpec("200 GbE IB-HDR", bandwidth_gbps=200, latency_us=2),
    "400GbE-IB": NetworkSpec("400 GbE IB-NDR", bandwidth_gbps=400, latency_us=1),
    "local-PCIe": NetworkSpec("Local PCIe", bandwidth_gbps=256, latency_us=0),

    # ── Measured on GCP us-central1-a, 2026-05-14 ──
    # Setup: c3-standard-44 (44 vCPU, gVNIC, NIC link 32 Gb/s) coord ↔
    # g2-standard-8 (8 vCPU, L4 GPU) worker. Intra-zone, no Tier_1.
    # iperf3 over TCP, 32 parallel streams to find the per-link cap.
    #
    # Effective per-VM egress is the bottleneck for these flows; ingress
    # is not VM-capped on GCP (only the receiver's NIC ceiling matters).
    # So a "link" name here is named after the *sender*'s VM family.
    "GCP-g2-default": NetworkSpec(
        # g2 (and other GPU instances) are not Tier_1-capable; their
        # sustained per-VM egress caps at ~10 Gb/s no matter the vCPU
        # count. Single TCP stream measured 9.66 Gb/s; 32 streams
        # aggregate 9.74 Gb/s — same ceiling, no scaling with N.
        "GCP g2 worker → coord (intra-zone, no Tier_1)",
        bandwidth_gbps=9.7, latency_us=100),
    "GCP-c3-default": NetworkSpec(
        # c3-standard-44 default (no Tier_1): NIC link is 32 Gb/s
        # (gVNIC, ethtool-confirmed). Measured 27.2 Gb/s sustained
        # egress from coord → worker with 32 streams (≈85% of NIC).
        # Ingress is uncapped at the VM level, so the same NIC ceiling
        # applies to the receive side.
        "GCP c3-standard-44 (default, gVNIC 32 Gb/s NIC)",
        bandwidth_gbps=32, latency_us=100),
    "GCP-c3-tier1": NetworkSpec(
        # c3-standard-44 with Tier_1 networking enabled (free flag,
        # gVNIC + 30+ vCPU required, no extra cost). Per GCP spec the
        # NIC ceiling rises to ~100 Gb/s for this size.
        "GCP c3-standard-44 + Tier_1 networking",
        bandwidth_gbps=100, latency_us=100),
}


@dataclass
class VerifyParams:
    freivalds_k: int = 10
    verify_every_n: int = 1
    max_workers: int = 4
    use_slalom: bool = True  # False = standard Freivalds (current), True = SLALOM (distributed plan)
    compression: str = "none"  # "none" or "fp16"
    tee_overhead_pct: float = 3.0


# ── Calibrated Cost Functions ───────────────────────────────────────────────
#
# These are calibrated from real Qwen3.5-9B benchmarks (RTX 5090, 18 cores).
# The key insight: CPU verification is memory-bandwidth-bound, not compute-bound.
# Standard Freivalds reads the full weight matrix per check (~192MB for 4096x12288).
# SLALOM reads only small vectors (s, s_tilde) that fit in cache.

# Calibration anchors from real benchmarks:
#   linear_freivalds: 56.14ms avg, weight size = H*I*4 bytes
#   For Qwen3.5-9B MLP: 4096*12288*4 = 192MB -> 56ms at 90 GB/s mem BW
#   Effective throughput: 192MB / 56ms = 3.4 GB/s (includes compute + overhead)
#   For Qwen3.5-9B Attention: 4096*4096*4 = 64MB -> ~18ms expected
#   matmul_freivalds: 13.52ms avg, operands Q@K^T are small in decode (seq=1)
#   D2H linear: 0.64ms for ~200KB tensors at ~312 MB/s effective
#   D2H matmul: 2.69ms for ~800KB tensors

# Calibrated constants:
FREIVALDS_EFFECTIVE_MEM_THROUGHPUT = 3.4e9  # bytes/sec at 90 GB/s DRAM, accounts for compute overhead
SLALOM_EFFECTIVE_MEM_THROUGHPUT = 40e9  # bytes/sec, cache-friendly small vectors
D2H_EFFECTIVE_THROUGHPUT = 312e6  # bytes/sec via PCIe pinned memory (with overhead)
GPU_COMPUTE_UTILIZATION = 0.35  # effective utilization for small batch inference


def _compression_ratio(compression: str) -> float:
    return 0.5 if compression == "fp16" else 1.0


def estimate_gpu_forward_ms(model: ModelSpec, gpu: GPUSpec, batch: int, seq_len: int, num_gpus: int) -> float:
    """Estimate GPU forward pass time for all layers."""
    H = model.hidden_size
    I = model.intermediate_size
    nh = model.num_attention_heads
    nkv = model.num_kv_heads
    hd = model.head_dim
    S = seq_len

    # Per-layer FLOPs
    # Attention: Q/K/V/O projections + Q@K^T + attn@V
    attn_proj_flops = 2 * batch * S * H * (H + 2 * nkv * hd + H)  # Q, K, V, O
    attn_matmul_flops = 2 * 2 * batch * nh * S * S * hd  # Q@K^T + attn@V
    attn_flops_per_layer = attn_proj_flops + attn_matmul_flops

    # MLP: gate, up, down projections
    mlp_flops_per_layer = 2 * batch * S * H * I * 3

    total_flops = (
        model.num_verified_attn_layers * attn_flops_per_layer
        + model.num_verified_mlp_layers * mlp_flops_per_layer
    )

    # For decode (seq=1), GPU is memory-bandwidth-bound (loading weights)
    # Model bytes that need to be read per forward: all weight parameters
    weight_read_bytes = model.param_bytes
    mem_bound_ms = (weight_read_bytes / num_gpus) / (gpu.memory_bw_gbps * 1e9) * 1000

    # Compute bound
    compute_bound_ms = (total_flops / num_gpus) / (gpu.bf16_tflops * 1e12 * GPU_COMPUTE_UTILIZATION) * 1000

    # GPU time = max(compute, memory) with pipeline parallel overhead
    base_ms = max(compute_bound_ms, mem_bound_ms)
    if num_gpus > 1:
        base_ms *= 1.15  # ~15% pipeline bubble
    return base_ms


def estimate_transfer_bytes(model: ModelSpec, batch: int, seq_len: int, compression: str) -> float:
    """Total bytes to transfer from GPU to CPU for verification."""
    H = model.hidden_size
    I = model.intermediate_size
    nh = model.num_attention_heads
    nkv = model.num_kv_heads
    hd = model.head_dim
    S = seq_len
    cr = _compression_ratio(compression)

    # Per attention layer: x + q_raw + k_raw + v_raw + scores + attn_out + o_raw
    attn_numel_per_layer = (
        batch * S * H  # x
        + batch * S * H  # q_raw
        + batch * S * nkv * hd  # k_raw
        + batch * S * nkv * hd  # v_raw
        + batch * nh * S * S  # scores (Q@K^T)
        + batch * nh * S * hd  # attn_out (P@V)
        + batch * S * H  # o_raw
    )

    # Per MLP layer: x + gate_raw + up_raw + down_raw
    mlp_numel_per_layer = (
        batch * S * H  # x
        + batch * S * I  # gate_raw
        + batch * S * I  # up_raw
        + batch * S * H  # down_raw
    )

    total_bytes = (
        model.num_verified_attn_layers * attn_numel_per_layer * 4 * cr
        + model.num_verified_mlp_layers * mlp_numel_per_layer * 4 * cr
    )
    return total_bytes


def estimate_linear_verify_ms_single(
    h_in: int, h_out: int, batch: int, seq_len: int,
    k: int, use_slalom: bool, cpu: CPUSpec,
) -> float:
    """Single linear layer verification time (1 worker thread)."""
    if use_slalom:
        # SLALOM: y@s + x@s_tilde, total data read:
        #   y: batch*seq*h_out*4, s: h_out*k*4
        #   x: batch*seq*h_in*4, s_tilde: h_in*k*4
        data_bytes = (batch * seq_len * h_out + h_out * k + batch * seq_len * h_in + h_in * k) * 4
        # Scale throughput by CPU memory bandwidth relative to benchmark CPU
        bw_scale = cpu.memory_bw_gbps / 90  # relative to bench CPU
        effective_throughput = SLALOM_EFFECTIVE_MEM_THROUGHPUT * min(bw_scale, 3.0)  # diminishing returns
        return (data_bytes / effective_throughput) * 1000
    else:
        # Standard Freivalds: reads weight matrix W[h_in, h_out]
        weight_bytes = h_in * h_out * 4
        data_bytes = weight_bytes + (batch * seq_len * (h_in + h_out) * k * 4)
        bw_scale = cpu.memory_bw_gbps / 90
        effective_throughput = FREIVALDS_EFFECTIVE_MEM_THROUGHPUT * min(bw_scale, 3.0)
        return (data_bytes / effective_throughput) * 1000


def estimate_matmul_verify_ms_single(
    m: int, n: int, p: int, batch_heads: int,
    k: int, cpu: CPUSpec,
) -> float:
    """Single dynamic matmul verification (Q@K^T or P@V). Always standard Freivalds."""
    # Reads: A[m,n], B[n,p], C[m,p], r[p,k]
    # For decode seq=1: very small tensors, cache-friendly
    data_bytes = (batch_heads * (m * n + n * p + m * p) + p * k) * 4
    bw_scale = cpu.memory_bw_gbps / 90
    effective_throughput = SLALOM_EFFECTIVE_MEM_THROUGHPUT * min(bw_scale, 3.0)  # small, cache-friendly
    return (data_bytes / effective_throughput) * 1000


def estimate_elementwise_ms_single(numel: int, cpu: CPUSpec) -> float:
    """Softmax/SiLU recompute on CPU."""
    data_bytes = numel * 4 * 2  # read input + write output
    bw_scale = cpu.memory_bw_gbps / 90
    effective_throughput = SLALOM_EFFECTIVE_MEM_THROUGHPUT * min(bw_scale, 3.0)
    return (data_bytes / effective_throughput) * 1000


# ── Full Forward Verification Cost ─────────────────────────────────────────

@dataclass
class VerifyCostBreakdown:
    linear_total_ms: float = 0  # total sequential linear verification
    matmul_total_ms: float = 0
    elementwise_total_ms: float = 0
    verify_total_sequential_ms: float = 0
    verify_total_parallel_ms: float = 0
    transfer_total_bytes: float = 0
    n_linear_checks: int = 0
    n_matmul_checks: int = 0
    n_elementwise_checks: int = 0


def estimate_full_verify_cost(
    model: ModelSpec, cpu: CPUSpec, batch: int, seq_len: int, vp: VerifyParams,
) -> VerifyCostBreakdown:
    """Estimate total CPU verification cost for one forward pass."""
    bc = VerifyCostBreakdown()
    H = model.hidden_size
    I = model.intermediate_size
    nh = model.num_attention_heads
    nkv = model.num_kv_heads
    hd = model.head_dim
    S = seq_len
    k = vp.freivalds_k

    # Linear checks
    # Attention: q_proj(H->H), k_proj(H->nkv*hd), v_proj(H->nkv*hd), o_proj(H->H)
    attn_linear_ms = (
        estimate_linear_verify_ms_single(H, H, batch, S, k, vp.use_slalom, cpu)  # q
        + estimate_linear_verify_ms_single(H, nkv * hd, batch, S, k, vp.use_slalom, cpu)  # k
        + estimate_linear_verify_ms_single(H, nkv * hd, batch, S, k, vp.use_slalom, cpu)  # v
        + estimate_linear_verify_ms_single(H, H, batch, S, k, vp.use_slalom, cpu)  # o
    )

    # MLP: gate(H->I), up(H->I), down(I->H)
    mlp_linear_ms = (
        estimate_linear_verify_ms_single(H, I, batch, S, k, vp.use_slalom, cpu)  # gate
        + estimate_linear_verify_ms_single(H, I, batch, S, k, vp.use_slalom, cpu)  # up
        + estimate_linear_verify_ms_single(I, H, batch, S, k, vp.use_slalom, cpu)  # down
    )

    bc.linear_total_ms = (
        model.num_verified_attn_layers * attn_linear_ms
        + model.num_verified_mlp_layers * mlp_linear_ms
    )
    bc.n_linear_checks = model.linear_checks_per_fwd

    # Matmul checks: Q@K^T[S,hd]@[hd,S], P@V[S,S]@[S,hd]
    qk_ms = estimate_matmul_verify_ms_single(S, hd, S, batch * nh, k, cpu)
    pv_ms = estimate_matmul_verify_ms_single(S, S, hd, batch * nh, k, cpu)
    bc.matmul_total_ms = model.num_verified_attn_layers * (qk_ms + pv_ms)
    bc.n_matmul_checks = model.matmul_checks_per_fwd

    # Elementwise: softmax over [batch, nh, S, S]
    softmax_numel = batch * nh * S * S
    bc.elementwise_total_ms = model.num_verified_attn_layers * estimate_elementwise_ms_single(softmax_numel, cpu)
    bc.n_elementwise_checks = model.elementwise_checks_per_fwd

    # Apply verify_every_n
    bc.linear_total_ms /= vp.verify_every_n
    bc.matmul_total_ms /= vp.verify_every_n
    bc.elementwise_total_ms /= vp.verify_every_n

    bc.verify_total_sequential_ms = bc.linear_total_ms + bc.matmul_total_ms + bc.elementwise_total_ms

    # Apply TEE overhead
    bc.verify_total_sequential_ms *= (1 + vp.tee_overhead_pct / 100)

    # Parallelization: limited by memory bandwidth contention
    # Each worker needs memory bandwidth. Effective parallelism limited by:
    # 1. Number of workers
    # 2. Memory bandwidth divided among workers (diminishing returns)
    max_parallel = min(vp.max_workers, cpu.cores)
    # Memory bandwidth scales sub-linearly: sqrt(workers) effective for BW-bound work
    # But compute-bound SLALOM scales nearly linearly
    if vp.use_slalom:
        effective_parallel = min(max_parallel, max(1, cpu.cores * 0.7))  # SLALOM is more cache-friendly
    else:
        # Standard Freivalds: BW-bound, sqrt scaling
        effective_parallel = min(max_parallel, max(1, math.sqrt(cpu.memory_bw_gbps / 10)))
    bc.verify_total_parallel_ms = bc.verify_total_sequential_ms / effective_parallel

    bc.transfer_total_bytes = estimate_transfer_bytes(model, batch, seq_len, vp.compression) / vp.verify_every_n

    return bc


# ── End-to-End Simulation ──────────────────────────────────────────────────

@dataclass
class SimResult:
    config_label: str = ""
    model_name: str = ""
    num_gpus: int = 0
    # Prefill phase (process full prompt)
    prefill_gpu_ms: float = 0
    prefill_transfer_ms: float = 0
    prefill_verify_ms: float = 0
    prefill_total_ms: float = 0
    prefill_tok_per_sec: float = 0
    # Decode phase (per token)
    decode_gpu_ms: float = 0
    decode_transfer_ms: float = 0
    decode_verify_ms: float = 0
    decode_total_ms: float = 0
    decode_tok_per_sec: float = 0
    # Aggregates
    gen_50tok_s: float = 0
    transfer_mb_per_fwd: float = 0
    overhead_vs_origin: float = 0
    bottleneck: str = ""
    oom: bool = False


def simulate(
    model: ModelSpec, gpu: GPUSpec, cpu: CPUSpec, net: NetworkSpec,
    num_gpus: int, batch: int, seq_len: int, vp: VerifyParams,
    gen_tokens: int = 50,
) -> SimResult:
    r = SimResult(model_name=model.name, num_gpus=num_gpus)

    # OOM check
    bytes_per_gpu = model.param_bytes / num_gpus
    if bytes_per_gpu > gpu.memory_gb * 1e9 * 0.85:
        r.oom = True
        r.bottleneck = f"OOM ({bytes_per_gpu/1e9:.0f}GB>{gpu.memory_gb*0.85:.0f}GB)"
        return r

    net_bytes_per_sec = net.bandwidth_gbps * 1e9 / 8

    # ── Prefill ──
    r.prefill_gpu_ms = estimate_gpu_forward_ms(model, gpu, batch, seq_len, num_gpus)

    prefill_verify = estimate_full_verify_cost(model, cpu, batch, seq_len, vp)
    r.prefill_verify_ms = prefill_verify.verify_total_parallel_ms

    # Transfer: each GPU sends its share in parallel
    total_xfer = prefill_verify.transfer_total_bytes
    xfer_per_gpu = total_xfer / num_gpus
    r.prefill_transfer_ms = (xfer_per_gpu / net_bytes_per_sec) * 1000
    r.prefill_transfer_ms += net.latency_us / 1000 * (model.linear_checks_per_fwd + model.matmul_checks_per_fwd) / vp.verify_every_n

    # Pipelined: max of the three stages
    r.prefill_total_ms = max(r.prefill_gpu_ms, r.prefill_transfer_ms, r.prefill_verify_ms)
    r.prefill_tok_per_sec = seq_len / (r.prefill_total_ms / 1000) if r.prefill_total_ms > 0 else 0

    # ── Decode (per token, seq_len=1) ──
    r.decode_gpu_ms = estimate_gpu_forward_ms(model, gpu, batch, 1, num_gpus)

    decode_verify = estimate_full_verify_cost(model, cpu, batch, 1, vp)
    r.decode_verify_ms = decode_verify.verify_total_parallel_ms

    decode_xfer_bytes = decode_verify.transfer_total_bytes
    decode_xfer_per_gpu = decode_xfer_bytes / num_gpus
    r.decode_transfer_ms = (decode_xfer_per_gpu / net_bytes_per_sec) * 1000
    r.decode_transfer_ms += net.latency_us / 1000 * (model.linear_checks_per_fwd + model.matmul_checks_per_fwd) / vp.verify_every_n

    # For decode, we flush each token, so total = max of stages
    # But if verify > gpu, we can't overlap perfectly — verify blocks next token's flush
    r.decode_total_ms = max(r.decode_gpu_ms, r.decode_transfer_ms, r.decode_verify_ms)

    r.decode_tok_per_sec = 1000 / r.decode_total_ms if r.decode_total_ms > 0 else 0
    r.gen_50tok_s = (r.prefill_total_ms + gen_tokens * r.decode_total_ms) / 1000
    r.transfer_mb_per_fwd = total_xfer / 1e6

    # Origin (unverified) for overhead calculation
    origin_decode_ms = estimate_gpu_forward_ms(model, gpu, batch, 1, 1)
    r.overhead_vs_origin = r.decode_total_ms / origin_decode_ms if origin_decode_ms > 0 else 0

    # Bottleneck
    stages = {
        "GPU": r.decode_gpu_ms,
        "Network": r.decode_transfer_ms,
        "CPU verify": r.decode_verify_ms,
    }
    r.bottleneck = max(stages, key=stages.get)

    return r


# ── Display ─────────────────────────────────────────────────────────────────

def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.1f}ms"


def print_table(results: List[SimResult], title: str) -> None:
    print(f"\n{'='*140}")
    print(f"  {title}")
    print(f"{'='*140}")
    hdr = (
        f"  {'Configuration':<50} "
        f"{'Prefill':>9} {'Dec/tok':>9} {'50tok':>8} {'tok/s':>7} "
        f"{'Xfer/fwd':>9} {'Overhead':>8} {'Bottleneck':<14}"
    )
    print(hdr)
    print(f"  {'-'*135}")

    for r in results:
        if r.config_label == "---":
            print(f"  {'-'*135}")
            continue
        if r.oom:
            print(f"  {r.config_label:<50} {'OOM: ' + r.bottleneck}")
            continue
        print(
            f"  {r.config_label:<50} "
            f"{_fmt_ms(r.prefill_total_ms):>9} "
            f"{_fmt_ms(r.decode_total_ms):>9} "
            f"{r.gen_50tok_s:>7.2f}s "
            f"{r.decode_tok_per_sec:>7.1f} "
            f"{r.transfer_mb_per_fwd:>8.1f}M "
            f"{r.overhead_vs_origin:>7.1f}x "
            f"{r.bottleneck:<14}"
        )


def print_detail(r: SimResult, label: str = "") -> None:
    if label:
        print(f"\n  {label}")
    if r.oom:
        print(f"    OOM: {r.bottleneck}")
        return
    print(f"    Prefill : GPU={_fmt_ms(r.prefill_gpu_ms):>8} | Xfer={_fmt_ms(r.prefill_transfer_ms):>8} | Verify={_fmt_ms(r.prefill_verify_ms):>8} | Total={_fmt_ms(r.prefill_total_ms):>8} ({r.prefill_tok_per_sec:.0f} tok/s)")
    print(f"    Decode  : GPU={_fmt_ms(r.decode_gpu_ms):>8} | Xfer={_fmt_ms(r.decode_transfer_ms):>8} | Verify={_fmt_ms(r.decode_verify_ms):>8} | Total={_fmt_ms(r.decode_total_ms):>8} ({r.decode_tok_per_sec:.1f} tok/s)")
    print(f"    50-token: {r.gen_50tok_s:.2f}s | Transfer: {r.transfer_mb_per_fwd:.1f} MB/fwd | Bottleneck: {r.bottleneck}")


# ── Experiments ─────────────────────────────────────────────────────────────

def sep() -> SimResult:
    r = SimResult()
    r.config_label = "---"
    return r


def run_calibration():
    """Compare model predictions vs real benchmark data."""
    print(f"\n{'='*140}")
    print(f"  CALIBRATION: Model Predictions vs Real Benchmark")
    print(f"  Qwen3.5-9B | RTX 5090 | 18 CPU cores | local D2H | standard Freivalds k=4")
    print(f"{'='*140}")

    model = MODELS["Qwen3.5-9B"]
    gpu = GPUS["RTX-5090"]
    cpu = CPUS["bench-18core"]
    net = NETWORKS["local-PCIe"]
    vp = VerifyParams(freivalds_k=4, verify_every_n=1, max_workers=4,
                      use_slalom=False, compression="none", tee_overhead_pct=0)

    r = simulate(model, gpu, cpu, net, 1, batch=1, seq_len=10, vp=vp)

    # Also compute with SLALOM for comparison
    vp_slalom = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=4,
                             use_slalom=True, compression="none", tee_overhead_pct=0)
    r_slalom = simulate(model, gpu, cpu, net, 1, batch=1, seq_len=10, vp=vp_slalom)

    print(f"\n  {'Metric':<40} {'Real':>12} {'Model(Frei)':>12} {'Model(SLALOM)':>14} {'Frei Ratio':>10}")
    print(f"  {'-'*90}")
    print(f"  {'Origin forward (ms)':<40} {'136.7':>12} {r.prefill_gpu_ms:>12.1f} {'--':>14} {r.prefill_gpu_ms/136.7:>10.2f}x")
    print(f"  {'Verified fwd (ms, local)':<40} {'1950':>12} {r.prefill_total_ms:>12.1f} {r_slalom.prefill_total_ms:>14.1f} {r.prefill_total_ms/1950:>10.2f}x")
    print(f"  {'Decode tok/s (verified)':<40} {'0.4':>12} {r.decode_tok_per_sec:>12.1f} {r_slalom.decode_tok_per_sec:>14.1f} {r.decode_tok_per_sec/0.4:>10.2f}x")
    print(f"  {'Decode per-token ms':<40} {'~2300':>12} {r.decode_total_ms:>12.1f} {r_slalom.decode_total_ms:>14.1f} {r.decode_total_ms/2300:>10.2f}x")
    print(f"  {'Bottleneck':<40} {'CPU verify':>12} {r.bottleneck:>12} {r_slalom.bottleneck:>14}")

    # Detailed breakdown
    vc = estimate_full_verify_cost(model, cpu, 1, 10, vp)
    vc_s = estimate_full_verify_cost(model, cpu, 1, 10, vp_slalom)
    print(f"\n  Verification breakdown (prefill, seq=10):")
    print(f"  {'':>4} {'Standard Freivalds':>20} {'SLALOM':>20}")
    print(f"  {'Linear (seq, ms)':<25} {vc.linear_total_ms:>20.1f} {vc_s.linear_total_ms:>20.1f}")
    print(f"  {'Matmul (seq, ms)':<25} {vc.matmul_total_ms:>20.1f} {vc_s.matmul_total_ms:>20.1f}")
    print(f"  {'Total sequential (ms)':<25} {vc.verify_total_sequential_ms:>20.1f} {vc_s.verify_total_sequential_ms:>20.1f}")
    print(f"  {'Total parallel (ms)':<25} {vc.verify_total_parallel_ms:>20.1f} {vc_s.verify_total_parallel_ms:>20.1f}")
    print(f"  {'Speedup':<25} {'1.0x':>20} {vc.verify_total_sequential_ms / max(0.001, vc_s.verify_total_sequential_ms):>19.0f}x")


def run_bandwidth_sweep():
    """Impact of network bandwidth across model sizes."""
    results = []
    for model_name in ["Qwen2.5-0.5B", "Qwen3.5-9B", "Llama-3-70B"]:
        model = MODELS[model_name]
        cpu = CPUS["EPYC-9654"]
        gpu = GPUS["H100-SXM"]
        num_gpus = max(1, math.ceil(model.param_bytes / (gpu.memory_gb * 1e9 * 0.85)))
        for net_name in ["1GbE", "GCP-g2-default", "10GbE", "25GbE",
                          "GCP-c3-default", "100GbE", "GCP-c3-tier1",
                          "200GbE-IB", "local-PCIe"]:
            net = NETWORKS[net_name]
            vp = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=64,
                              use_slalom=True, compression="fp16", tee_overhead_pct=3)
            r = simulate(model, gpu, cpu, net, num_gpus, batch=1, seq_len=32, vp=vp)
            r.config_label = f"{model_name} {num_gpus}xH100 | {net.name}"
            results.append(r)
        results.append(sep())

    print_table(results, "BANDWIDTH SWEEP: SLALOM + fp16 compression | EPYC 9654 96C | batch=1 seq=32")


def run_gpu_scaling():
    """More GPUs feeding one CPU TEE."""
    results = []
    for model_name in ["Qwen3.5-9B", "Llama-3-70B"]:
        model = MODELS[model_name]
        cpu = CPUS["EPYC-9654"]
        net = NETWORKS["100GbE"]
        for ng in [1, 2, 4, 8]:
            gpu = GPUS["H100-SXM"]
            vp = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=64,
                              use_slalom=True, compression="fp16", tee_overhead_pct=3)
            r = simulate(model, gpu, cpu, net, ng, batch=1, seq_len=32, vp=vp)
            r.config_label = f"{model_name} {ng}xH100 | 100GbE | EPYC-9654"
            results.append(r)
        results.append(sep())

    print_table(results, "GPU SCALING: Pipeline Parallel | SLALOM | EPYC 9654")


def run_cpu_comparison():
    """How CPU choice affects verification throughput."""
    results = []
    model = MODELS["Qwen3.5-9B"]
    gpu = GPUS["H100-SXM"]
    net = NETWORKS["100GbE"]

    for cpu_name in ["i9-14900K", "Xeon-8480+", "EPYC-9654", "EPYC-9754", "2xEPYC-9754"]:
        cpu = CPUS[cpu_name]
        vp = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=cpu.cores,
                          use_slalom=True, compression="fp16", tee_overhead_pct=3)
        r = simulate(model, gpu, cpu, net, 1, batch=1, seq_len=32, vp=vp)
        r.config_label = f"Qwen3.5-9B 1xH100 | 100GbE | {cpu.name}"
        results.append(r)

    results.append(sep())

    # Same but with Standard Freivalds (current implementation)
    for cpu_name in ["i9-14900K", "Xeon-8480+", "EPYC-9654", "2xEPYC-9754"]:
        cpu = CPUS[cpu_name]
        vp = VerifyParams(freivalds_k=4, verify_every_n=1, max_workers=cpu.cores,
                          use_slalom=False, compression="none", tee_overhead_pct=3)
        r = simulate(model, gpu, cpu, net, 1, batch=1, seq_len=32, vp=vp)
        r.config_label = f"[Freivalds] 1xH100 | 100GbE | {cpu.name}"
        results.append(r)

    print_table(results, "CPU COMPARISON: Qwen3.5-9B | SLALOM vs Standard Freivalds")


def run_verify_tradeoffs():
    """Impact of k and verify_every_n."""
    results = []
    model = MODELS["Qwen3.5-9B"]
    gpu = GPUS["H100-SXM"]
    cpu = CPUS["EPYC-9654"]
    net = NETWORKS["100GbE"]

    configs = [
        (10, 1, "k=10 every=1 (full)"),
        (4, 1, "k=4  every=1"),
        (10, 2, "k=10 every=2"),
        (4, 2, "k=4  every=2"),
        (10, 4, "k=10 every=4"),
        (4, 4, "k=4  every=4"),
        (2, 8, "k=2  every=8 (min)"),
    ]
    for k, en, desc in configs:
        vp = VerifyParams(freivalds_k=k, verify_every_n=en, max_workers=64,
                          use_slalom=True, compression="fp16", tee_overhead_pct=3)
        r = simulate(model, gpu, cpu, net, 1, batch=1, seq_len=32, vp=vp)
        error_prob = 1 - (1 - 2**(-k))**(1.0/en)
        r.config_label = f"{desc}  err_prob={error_prob:.4f}"
        results.append(r)

    print_table(results, "VERIFICATION TRADEOFFS: Qwen3.5-9B | 1xH100 | EPYC-9654 | 100GbE")


def run_multi_model():
    """Different models on different GPUs -> 1 CPU TEE."""
    print(f"\n{'='*140}")
    print(f"  MULTI-MODEL SCENARIO: 3 Different Models -> 1 CPU TEE (EPYC 9654, 100GbE)")
    print(f"{'='*140}")

    cpu = CPUS["EPYC-9654"]
    net = NETWORKS["100GbE"]
    vp = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=32,
                      use_slalom=True, compression="fp16", tee_overhead_pct=3)

    scenarios = [
        ("GPU1: Qwen3.5-9B (H100)", MODELS["Qwen3.5-9B"], GPUS["H100-SXM"], 1),
        ("GPU2: Llama-3.2-1B (4090)", MODELS["Llama-3.2-1B"], GPUS["RTX-4090"], 1),
        ("GPU3: Qwen2.5-0.5B (4090)", MODELS["Qwen2.5-0.5B"], GPUS["RTX-4090"], 1),
    ]

    total_cpu_verify_decode = 0
    total_xfer_decode = 0

    for desc, model, gpu, ng in scenarios:
        r = simulate(model, gpu, cpu, net, ng, batch=1, seq_len=32, vp=vp)
        print_detail(r, desc)
        dc = estimate_full_verify_cost(model, cpu, 1, 1, vp)
        total_cpu_verify_decode += dc.verify_total_sequential_ms
        total_xfer_decode += dc.transfer_total_bytes

    print(f"\n  --- Aggregate CPU TEE Load (decode, per token) ---")
    print(f"  Total CPU verify (sequential): {total_cpu_verify_decode:.2f} ms")
    effective_par = min(96, 96 * 0.7)
    print(f"  Total CPU verify (parallel {effective_par:.0f} workers): {total_cpu_verify_decode / effective_par:.2f} ms")
    print(f"  Total network ingest: {total_xfer_decode / 1e3:.1f} KB/token")
    cpu_headroom = max(0, (1 - total_cpu_verify_decode / effective_par / 5) * 100)
    print(f"  CPU headroom estimate: {cpu_headroom:.0f}% (can add more GPUs)")


def run_large_model_cluster():
    """70B model across GPU cluster."""
    results = []
    cpu = CPUS["EPYC-9654"]
    vp = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=64,
                      use_slalom=True, compression="fp16", tee_overhead_pct=3)

    configs = [
        ("2xA100-80G 100GbE", GPUS["A100-80G"], NETWORKS["100GbE"], 2),
        ("4xA100-80G 100GbE", GPUS["A100-80G"], NETWORKS["100GbE"], 4),
        ("2xH100 100GbE", GPUS["H100-SXM"], NETWORKS["100GbE"], 2),
        ("4xH100 100GbE", GPUS["H100-SXM"], NETWORKS["100GbE"], 4),
        ("8xH100 200GbE-IB", GPUS["H100-SXM"], NETWORKS["200GbE-IB"], 8),
        ("2xH200 100GbE", GPUS["H200"], NETWORKS["100GbE"], 2),
        ("2xB200 200GbE-IB", GPUS["B200"], NETWORKS["200GbE-IB"], 2),
        ("4xB200 400GbE-IB", GPUS["B200"], NETWORKS["400GbE-IB"], 4),
    ]

    for desc, gpu, net, ng in configs:
        r = simulate(MODELS["Llama-3-70B"], gpu, cpu, net, ng, batch=1, seq_len=32, vp=vp)
        r.config_label = f"Llama-3-70B {desc}"
        results.append(r)

    print_table(results, "LARGE MODEL CLUSTER: Llama-3-70B | EPYC 9654 | SLALOM")


def run_tdx_4090_detailed():
    """Detailed breakdown: TDX Xeon CPU + multiple RTX 4090 GPUs, home vs cluster bandwidth."""
    print(f"\n{'='*140}")
    print(f"  TDX CPU TEE + Multiple RTX 4090 GPUs — Detailed Breakdown")
    print(f"  Home bandwidth (1GbE, WiFi-6, 2.5GbE, 10GbE) vs Cluster bandwidth (25GbE, 100GbE)")
    print(f"{'='*140}")

    gpu = GPUS["RTX-4090"]
    # TDX-capable Xeon: Sapphire Rapids / Emerald Rapids
    # Xeon 8480+ (56C) is a common TDX-capable server CPU
    cpu_tdx = CPUS["Xeon-8480+"]

    home_nets = {
        "1GbE (home ethernet)": NetworkSpec("1 GbE", bandwidth_gbps=1, latency_us=500),
        "WiFi-6 (~1.2Gbps real)": NetworkSpec("WiFi-6", bandwidth_gbps=1.2, latency_us=2000),
        "2.5GbE (home upgrade)": NetworkSpec("2.5 GbE", bandwidth_gbps=2.5, latency_us=300),
        "10GbE (prosumer NIC)": NetworkSpec("10 GbE", bandwidth_gbps=10, latency_us=100),
    }
    cluster_nets = {
        "25GbE (small cluster)": NetworkSpec("25 GbE", bandwidth_gbps=25, latency_us=50),
        "100GbE (datacenter)": NetworkSpec("100 GbE", bandwidth_gbps=100, latency_us=10),
    }

    models_to_test = ["Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen3.5-9B"]

    # ── Part 1: Single model, sweep bandwidth and GPU count ──
    for model_name in models_to_test:
        model = MODELS[model_name]
        max_4090 = max(1, math.ceil(model.param_bytes / (gpu.memory_gb * 1e9 * 0.85)))
        if max_4090 > 1:
            gpu_counts = [max_4090, max_4090 * 2]
        else:
            gpu_counts = [1, 2, 4]

        print(f"\n  ┌─ {model_name} (H={model.hidden_size}, I={model.intermediate_size}, L={model.num_layers})")
        print(f"  │  Params: {model.param_bytes/1e9:.1f}GB bf16 | Min GPUs: {max_4090} (4090 24GB)")
        print(f"  │  Verified layers: {model.num_verified_attn_layers} attn + {model.num_verified_mlp_layers} MLP")
        print(f"  │  Checks/fwd: {model.linear_checks_per_fwd} linear + {model.matmul_checks_per_fwd} matmul")

        # Show transfer size
        vp_s = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=56,
                            use_slalom=True, compression="fp16", tee_overhead_pct=3)
        xfer_prefill = estimate_transfer_bytes(model, 1, 32, "fp16")
        xfer_decode = estimate_transfer_bytes(model, 1, 1, "fp16")
        print(f"  │  Transfer/fwd: prefill(seq=32)={xfer_prefill/1e6:.1f}MB  decode(seq=1)={xfer_decode/1e3:.1f}KB")

        # Verification cost detail
        vc_pre = estimate_full_verify_cost(model, cpu_tdx, 1, 32, vp_s)
        vc_dec = estimate_full_verify_cost(model, cpu_tdx, 1, 1, vp_s)
        print(f"  │  CPU verify (SLALOM, 56C Xeon TDX):")
        print(f"  │    Prefill: {vc_pre.verify_total_sequential_ms:.2f}ms seq -> {vc_pre.verify_total_parallel_ms:.2f}ms parallel")
        print(f"  │    Decode:  {vc_dec.verify_total_sequential_ms:.3f}ms seq -> {vc_dec.verify_total_parallel_ms:.4f}ms parallel")

        # Also show standard Freivalds for comparison
        vp_f = VerifyParams(freivalds_k=4, verify_every_n=1, max_workers=56,
                            use_slalom=False, compression="none", tee_overhead_pct=3)
        vc_f_dec = estimate_full_verify_cost(model, cpu_tdx, 1, 1, vp_f)
        print(f"  │  CPU verify (Freivalds k=4, for comparison):")
        print(f"  │    Decode:  {vc_f_dec.verify_total_sequential_ms:.1f}ms seq -> {vc_f_dec.verify_total_parallel_ms:.1f}ms parallel")

        for ng in gpu_counts:
            if model.param_bytes / ng > gpu.memory_gb * 1e9 * 0.85:
                continue

            print(f"  │")
            print(f"  ├─ {ng}x RTX 4090 (pipeline parallel)")

            gpu_prefill = estimate_gpu_forward_ms(model, gpu, 1, 32, ng)
            gpu_decode = estimate_gpu_forward_ms(model, gpu, 1, 1, ng)
            print(f"  │  GPU forward: prefill={gpu_prefill:.1f}ms  decode={gpu_decode:.1f}ms")

            # Home bandwidth
            print(f"  │")
            print(f"  │  ┌─ HOME BANDWIDTH ─────────────────────────────────────────────────────────────────────")
            print(f"  │  │  {'Network':<28} {'Prefill':>9} {'Dec/tok':>9} {'50tok':>8} {'tok/s':>7} {'Bottleneck':<15} {'Transfer':>9}")
            print(f"  │  │  {'-'*92}")
            for net_label, net in home_nets.items():
                r = simulate(model, gpu, cpu_tdx, net, ng, batch=1, seq_len=32, vp=vp_s)
                print(f"  │  │  {net_label:<28} {_fmt_ms(r.prefill_total_ms):>9} {_fmt_ms(r.decode_total_ms):>9} {r.gen_50tok_s:>7.2f}s {r.decode_tok_per_sec:>7.1f} {r.bottleneck:<15} {r.transfer_mb_per_fwd:>7.1f}MB")
            print(f"  │  └─")

            # Cluster bandwidth
            print(f"  │  ┌─ CLUSTER BANDWIDTH ──────────────────────────────────────────────────────────────────")
            print(f"  │  │  {'Network':<28} {'Prefill':>9} {'Dec/tok':>9} {'50tok':>8} {'tok/s':>7} {'Bottleneck':<15} {'Transfer':>9}")
            print(f"  │  │  {'-'*92}")
            for net_label, net in cluster_nets.items():
                r = simulate(model, gpu, cpu_tdx, net, ng, batch=1, seq_len=32, vp=vp_s)
                print(f"  │  │  {net_label:<28} {_fmt_ms(r.prefill_total_ms):>9} {_fmt_ms(r.decode_total_ms):>9} {r.gen_50tok_s:>7.2f}s {r.decode_tok_per_sec:>7.1f} {r.bottleneck:<15} {r.transfer_mb_per_fwd:>7.1f}MB")
            print(f"  │  └─")

        print(f"  └─")

    # ── Part 2: Multi-model on mixed 4090s ──
    print(f"\n  {'='*120}")
    print(f"  MULTI-MODEL: 3 x RTX 4090, each running a different model -> 1 Xeon TDX")
    print(f"  {'='*120}")

    multi_scenarios = [
        ("4090 #1: Qwen2.5-0.5B", MODELS["Qwen2.5-0.5B"]),
        ("4090 #2: Llama-3.2-1B", MODELS["Llama-3.2-1B"]),
        ("4090 #3: Qwen3.5-9B", MODELS["Qwen3.5-9B"]),
    ]

    for net_label, net in [("10GbE (home)", NetworkSpec("10GbE", 10, 100)),
                           ("100GbE (cluster)", NetworkSpec("100GbE", 100, 10))]:
        print(f"\n  Network: {net_label}")
        total_cpu_dec = 0
        total_xfer_dec = 0
        for desc, model in multi_scenarios:
            r = simulate(model, gpu, cpu_tdx, net, 1, batch=1, seq_len=32, vp=vp_s)
            dc = estimate_full_verify_cost(model, cpu_tdx, 1, 1, vp_s)
            total_cpu_dec += dc.verify_total_sequential_ms
            total_xfer_dec += dc.transfer_total_bytes
            print(f"    {desc:<30} decode={_fmt_ms(r.decode_total_ms):>8} ({r.decode_tok_per_sec:.1f} tok/s)  bottleneck={r.bottleneck}")

        effective_par = min(56, 56 * 0.7)
        print(f"    ── Aggregate CPU TEE: {total_cpu_dec:.2f}ms seq / {total_cpu_dec/effective_par:.3f}ms parallel | {total_xfer_dec/1e3:.1f} KB/tok network")

    # ── Part 3: Recommendations ──
    print(f"\n  {'='*120}")
    print(f"  RECOMMENDATIONS for TDX + 4090 Setup")
    print(f"  {'='*120}")
    print(f"""
  WITH SLALOM (planned distributed architecture):
    - CPU verification is negligible (<0.1ms/token) — bottleneck is ALWAYS GPU or network
    - Home 1GbE:  Works for 0.5B models (9 tok/s). Too slow for 9B+ (prefill takes seconds)
    - Home 10GbE: Sweet spot for home setup. 0.5B: 45 tok/s, 1B: 61 tok/s, 9B: limited by GPU
    - Cluster 25GbE+: Network stops being bottleneck. GPU becomes the limiting factor
    - 4090 is memory-BW limited at 1008 GB/s → decode ~17ms/tok for 9B (GPU-bound)
    - More 4090s help for large models (pipeline parallel) but not for small ones

  WITHOUT SLALOM (current standard Freivalds):
    - CPU verification dominates: 56C Xeon → 300-2000ms per token for 9B models
    - Adding GPUs doesn't help — CPU is the bottleneck regardless of network
    - SLALOM is a prerequisite for practical distributed deployment

  HARDWARE RECOMMENDATIONS:
    CPU TEE:  Xeon 8480+ (56C) or newer with TDX. Even 2nd-tier Xeon works with SLALOM
    Network:  10GbE minimum for home (< $100 NIC). 25GbE+ for cluster
    GPUs:     1x 4090 per small model (<7B). 2+ for 9B pipeline parallel
    Upgrade:  If network-bound, 2.5GbE->10GbE gives 4x improvement for ~$80
""")


def run_slalom_vs_freivalds():
    """Direct comparison of SLALOM vs standard Freivalds across models."""
    results = []
    cpu = CPUS["EPYC-9654"]
    gpu = GPUS["H100-SXM"]
    net = NETWORKS["100GbE"]

    for model_name in ["Qwen2.5-0.5B", "Qwen3.5-9B", "Llama-3-70B"]:
        model = MODELS[model_name]
        ng = max(1, math.ceil(model.param_bytes / (gpu.memory_gb * 1e9 * 0.85)))

        # Standard Freivalds
        vp_f = VerifyParams(freivalds_k=4, verify_every_n=1, max_workers=64,
                            use_slalom=False, compression="none", tee_overhead_pct=3)
        r_f = simulate(model, gpu, cpu, net, ng, batch=1, seq_len=32, vp=vp_f)
        r_f.config_label = f"[Freivalds k=4] {model_name} {ng}xH100"
        results.append(r_f)

        # SLALOM
        vp_s = VerifyParams(freivalds_k=10, verify_every_n=1, max_workers=64,
                            use_slalom=True, compression="fp16", tee_overhead_pct=3)
        r_s = simulate(model, gpu, cpu, net, ng, batch=1, seq_len=32, vp=vp_s)
        r_s.config_label = f"[SLALOM k=10]   {model_name} {ng}xH100"
        results.append(r_s)

        results.append(sep())

    print_table(results, "SLALOM vs STANDARD FREIVALDS: Impact of Preprocessed Verification")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("  DecenTrain Distributed Verification Performance Model")
    print("  1 CPU TEE + N GPU Workers — Latency & Throughput Predictions")
    print("  Two verification modes: Standard Freivalds (current) vs SLALOM (planned)")
    print("=" * 140)

    run_calibration()
    run_tdx_4090_detailed()
    run_slalom_vs_freivalds()
    run_bandwidth_sweep()
    run_gpu_scaling()
    run_cpu_comparison()
    run_verify_tradeoffs()
    run_multi_model()
    run_large_model_cluster()

    print(f"\n{'='*140}")
    print("  KEY TAKEAWAYS:")
    print("  1. SLALOM preprocessing is critical: ~100-200x faster than standard Freivalds")
    print("  2. With SLALOM, CPU verification is fast enough that network becomes the bottleneck")
    print("  3. 100GbE is recommended; 10GbE works for small models; 1GbE is too slow for >1B")
    print("  4. CPU core count matters for standard Freivalds; less so for SLALOM (cache-friendly)")
    print("  5. For 70B models: 4xH100 + EPYC-9654 + 100GbE gives practical throughput")
    print("  6. verify_every_n=4 cuts overhead ~4x with acceptable confidence tradeoff")
    print("  7. fp16 compression halves transfer cost with negligible MSE impact")
    print(f"{'='*140}")


if __name__ == "__main__":
    main()
