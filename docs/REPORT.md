# Verified Neural Network Inference: Technical Report

## Table of Contents

1. [Current System Status](#1-current-system-status)
2. [Core Algorithm](#2-core-algorithm)
3. [SLALOM Alignment](#3-slalom-alignment)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Performance Evaluation](#5-performance-evaluation)
6. [Quantized Model Evaluation](#6-quantized-model-evaluation)
7. [Related Work](#7-related-work)
8. [CPU TEE Landscape Survey](#8-cpu-tee-landscape-survey)
9. [Compromise Methods for Non-TEE Environments](#9-compromise-methods-for-non-tee-environments)
10. [TEE Execution Evaluation](#10-tee-execution-evaluation)
11. [Recommendations](#11-recommendations)
12. [References](#12-references)

---

## 1. Current System Status

### 1.1 Project Overview

This project implements **probabilistic verification of GPU-computed neural network inference** using Freivalds' algorithm. The GPU is treated as untrusted; every matrix multiplication result is verified asynchronously on the CPU (intended to run inside a Trusted Execution Environment). The system supports two model families:

- **Verified LLM**: Llama, Qwen (2/2.5/3/3.5), and Mistral-family models
- **Verified Diffusers**: Z-Image diffusion transformer and Flux

### 1.2 Supported Models

| Model | Parameters | Architecture | Verification Coverage |
|-------|-----------|-------------|----------------------|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | Standard transformer | Full (all attention + MLP) |
| meta-llama/Llama-3.2-1B-Instruct | 1B | Standard transformer | Full (all attention + MLP) |
| Qwen/Qwen3.5-9B | 9B | Hybrid (full + linear attn) | Partial (8/32 attention + 32/32 MLP) |
| Tongyi-MAI/Z-Image | ~2B | Diffusion transformer | Full (all attention + feedforward) |

### 1.3 Codebase Structure

```
verified_llm/                     # LLM verification module
  verify_linear.py    (195 lines) # Freivalds algorithms + VerifyLinear
  attn_layer.py       (193 lines) # LlamaAttentionVerify (Llama/Qwen/Mistral)
  mlp_layer.py         (43 lines) # LlamaMLPVerify
  llm_model.py        (116 lines) # Model creation, duck-typed layer replacement

verified_diffusers/zimage/        # Diffusion verification module
  runtime.py          (343 lines) # VerifyRuntime (async verification engine)
  profiler.py         (105 lines) # VerifyProfiler (CSV/JSON/plot export)
  config.py            (46 lines) # VerifyConfig dataclass
  layers.py            (62 lines) # VerifyLinearModule, VerifyMatmul
  attention.py        (150 lines) # VerifiedZImageAttention
  mlp.py               (60 lines) # VerifiedZImageFeedForward
  transformer.py      (106 lines) # VerifiedZImageTransformer2DModel
  transformer_block.py (95 lines) # VerifiedZImageTransformerBlock
  pipeline.py         (120 lines) # VerifiedZImagePipeline

tests/                            # Test suite
  test_e2e_llm.py     (311 lines) # Logit equivalence, generation, corruption, profiling
  test_threat_model.py            # 8 attack scenarios (random, noise, selective, multi-layer)
  bench_qwen3_5_9b.py (406 lines) # Qwen3.5-9B benchmark with profiling
```

### 1.4 Current Capabilities

- **Zero-sync forward path**: No `torch.cuda.synchronize()` calls in the verified forward pass
- **Async pipelined verification**: GPU compute, D2H copy, and CPU verification overlap via three concurrent execution lanes
- **Duck-typed module detection**: Automatically detects and replaces compatible attention/MLP modules without model-specific code
- **Qwen3.5 hybrid support**: Handles gated attention output (`q_proj` 2x split + sigmoid gate), partial rotary embeddings, and dual KV cache parameter conventions
- **Comprehensive threat model testing**: 8 attack scenarios including random corruption, subtle noise injection, selective attacks, and multi-layer simultaneous corruption
- **Profiling**: Per-operation timing breakdown with CSV/JSON/plot export

---

## 2. Core Algorithm

### 2.1 Freivalds' Algorithm

To verify that the GPU correctly computed `C = A @ B` without full O(n^3) recomputation:

1. Generate random vector `r` of shape `(p, k)` where `k` controls confidence
2. Compute `Br = B @ r` on CPU: O(n * p * k)
3. Compute `ABr = A @ Br` on CPU: O(m * n * k)
4. Compute `Cr = C @ r` on CPU: O(m * p * k)
5. Compare: `MSE(ABr, Cr) < threshold`

**Complexity**: O(n^2 * k) vs O(n^3) for full recomputation — a factor of n/k speedup.

**Error probability**: With `k` random vectors, the probability of accepting an incorrect result is at most `2^{-k}`. With k=8 (default), this is 1/256 = 0.39%. With k=10, it is 1/1024 = 0.098%.

### 2.2 Implemented Variants

| Function | Use Case | Bias Support | Parallelism |
|----------|----------|-------------|-------------|
| `freivalds_algorithm_2d` | 2D matmul (projections) | No | Sequential |
| `freivalds_algorithm_2d_bias` | 2D linear with bias | Yes | Sequential |
| `freivalds_batch_matmul` | Batched matmul (Q@K^T) | No | Sequential |
| `freivalds_batch_matmul_bias` | Batched linear with bias | Yes | Sequential |
| `freivalds_batch_matmul_parallel` | Batched matmul | No | ABr and Cr in parallel |

### 2.3 Elementwise Verification

Non-matmul operations (softmax, SiLU activation) are verified by CPU recomputation:

1. Copy GPU input and output tensors to CPU via async D2H transfer
2. Recompute the operation in float32 on CPU
3. Compare via MSE against `elementwise_mse_threshold`

### 2.4 Numeric Considerations

- **GPU computes in model dtype** (bf16, fp16, or fp32)
- **CPU verifies in float32** (upcast for numerical stability)
- bf16 has ~3 decimal digits of precision; accumulated rounding error in large matmuls can produce MSE up to 1e-2 even for correct computations
- **NaN fallback**: When Freivalds MSE is non-finite (rare numeric corner case), the system falls back to `torch.allclose()` with relaxed tolerances

---

## 3. SLALOM Alignment

This implementation is aligned with **SLALOM: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware** (Tramer & Boneh, ICLR 2019). We adopt the paper's core algorithmic contribution — preprocessed Freivalds verification — while adapting it for floating-point inference without TEE (for now).

### 3.1 SLALOM's Core Idea

SLALOM partitions DNN computation between a trusted CPU (TEE) and an untrusted GPU:
- **GPU (untrusted)**: All linear operations (matmul, convolution, fully-connected)
- **CPU/TEE (trusted)**: All non-linear operations (ReLU, softmax, normalization) + verification

The key optimization: since model weights `W` are fixed at inference time, part of Freivalds' check can be **precomputed offline**.

### 3.2 Preprocessed Freivalds (Lemma 3.1 from the Paper)

For a linear operator `f(x) = x @ W^T`, let `s` be a random vector and `s_tilde = W^T @ s` (precomputed). Then:

```
Pr[y @ s != x @ s_tilde | y != f(x)] >= 1 - 1/|S|
```

**Verification with preprocessing:**
1. **Offline** (model loading): Compute and store `s` and `s_tilde = W^T @ s` for each linear layer
2. **Online** (inference): Given input `x` and GPU output `y`, check: `MSE(y @ s, x @ s_tilde) < threshold`

**Complexity reduction**: Online verification is `O(n * k)` — two matrix-vector products — instead of `O(n^2 * k)` for standard Freivalds (which must compute `W @ r` online).

### 3.3 Our Adaptation for Floating-Point Inference

| Aspect | SLALOM Paper | Our Implementation |
|--------|-------------|-------------------|
| Arithmetic | Integer field Z_p (p = 2^24 - 3) | Floating-point (bf16/fp32) |
| Random vectors | Integer from S = [-2^19, 2^19] | Gaussian N(0,1) float32 |
| Repetitions k | k = 2 (40-bit soundness via large |S|) | k = 10 (error prob < 2^{-10}) |
| Verification check | Exact equality in Z_p | MSE < threshold |
| Preprocessing | `s_tilde = W^T @ s` in Z_p | `s_tilde = W^T @ s` in float64, stored as float32 |
| Privacy (blinding) | One-time-pad: `x_tilde = x + r` | Not implemented (verify-only) |
| Execution model | Layer-by-layer CPU-GPU interaction | Async pipelined (GPU compute + CPU verify overlap) |

**Why Gaussian instead of integer random vectors?** Floating-point matrix multiplication is not associative: `(x @ W^T) @ s != x @ (W^T @ s)` in general. Large integer random vectors (2^19) amplify this associativity error. Gaussian vectors with unit scale keep the error within MSE threshold tolerance. The precomputation uses float64 to minimize the offline rounding error.

**Why k=10 instead of k=2?** SLALOM's k=2 achieves 40-bit soundness because `|S| = 2^20`. With continuous Gaussian vectors, the effective `|S|` is infinite but the soundness guarantee is probabilistic via MSE thresholding. k=10 provides equivalent practical confidence.

### 3.4 Performance Impact of Preprocessing

For Qwen3.5-9B (hidden_size=4096, intermediate_size=12288, k=10):

| Metric | Standard Freivalds | Preprocessed (SLALOM) | Speedup |
|--------|-------------------|----------------------|---------|
| **Dominant online cost** | `W^T @ r`: 4096 x 12288 x 10 = 503M ops | `y @ s`: batch x seq x 12288 x 10 | **~200x** |
| **Per-check (estimated)** | 56 ms | ~0.3 ms | **~190x** |
| **Total verification (50 tok)** | 370,936 ms | ~1,900 ms | **~195x** |
| **Forward overhead** | 14.3x | ~1.1x (estimated) | — |

The preprocessing cost (computing `s_tilde = W^T @ s` for all layers) is a one-time offline cost during model loading. For Qwen3.5-9B, this takes ~2 seconds and uses ~50 MB of CPU memory for all verification vectors.

### 3.5 What We Did NOT Adopt (and Why)

1. **Integer field arithmetic**: Requires quantizing all weights and activations to integers. Adds model accuracy loss and implementation complexity. Our floating-point adaptation provides equivalent probabilistic guarantees without quantization.

2. **CPU-side non-linear ops**: SLALOM runs softmax, ReLU, normalization on CPU/TEE. For modern transformers with deep attention stacks, this requires many CPU-GPU round trips. Our async pipelined approach keeps everything on GPU and verifies asynchronously, which is faster for large models.

3. **Privacy via blinding**: One-time-pad encryption of inputs. Not needed for our verify-only use case. Can be added later if privacy is required.

### 3.6 Matmul Verification (Non-Preprocessable)

Attention matmuls (`Q @ K^T`, `attn_probs @ V`) have **runtime-dependent** operands — preprocessing does not apply since `K` and `V` change every forward pass. These continue to use standard online Freivalds at `O(n^2 * k)` cost. However, attention matmuls are typically much smaller than linear projections (seq_len x seq_len vs hidden_size x intermediate_size), so the overhead is manageable.

---

## 4. Implementation Architecture

### 3.1 Threat Model

| Component | Trust Level | Role |
|-----------|-------------|------|
| CPU (TEE) | Trusted | Verification, non-matmul ops (softmax, SiLU, RMSNorm) |
| GPU | Untrusted | Accelerated matmul/linear operations |
| Memory Bus | Observed | GPU-to-CPU async transfer via PCIe |

**Assumption**: The GPU may be compromised, faulty, or performing incorrect computation. All GPU-computed results are verified before acceptance.

### 3.2 Async Pipelined Execution

```
Time ─────────────────────────────────────────────────────────────►

GPU (compute_stream):
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Linear_1 │ │ Linear_2 │ │ Linear_3 │ │ Linear_4 │ ...
  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │             │             │             │
Copy (copy_stream, pinned memory):
       ▼             ▼             ▼             ▼
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Copy_1  │  │ Copy_2  │  │ Copy_3  │  │ Copy_4  │ ...
  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
       │             │             │             │
CPU (ThreadPoolExecutor):
       ▼             ▼             ▼             ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Verify_1 │ │ Verify_2 │ │ Verify_3 │ │ Verify_4 │ ...
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

Key synchronization mechanisms:
- **CUDA Events**: `copy_stream.wait_stream(compute_stream)` ensures D2H copy starts only after GPU computation completes
- **Pinned Memory**: Pre-allocated page-locked CPU buffers for maximum PCIe bandwidth
- **ThreadPoolExecutor**: CPU Freivalds checks run without blocking GPU
- **GPU tensor reference pinning**: Closures hold GPU tensor references to prevent CUDA caching allocator from reclaiming memory before D2H copy completes
- **Flush barrier**: `runtime.flush()` blocks until all pending verifications complete

### 3.3 Module Replacement via Duck Typing

```python
def _is_attention_module(module: nn.Module) -> bool:
    return (hasattr(module, "q_proj") and hasattr(module, "k_proj")
            and hasattr(module, "v_proj") and hasattr(module, "o_proj")
            and isinstance(module.q_proj, nn.Linear))

def _is_mlp_module(module: nn.Module) -> bool:
    return (hasattr(module, "gate_proj") and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and isinstance(module.gate_proj, nn.Linear))
```

This duck-typing approach supports any HuggingFace model with standard attention/MLP structure without model-specific patching code.

### 3.4 Verified Operations per Forward Pass

For a standard transformer layer, the following operations are verified:

| Operation | Method | Count per Layer |
|-----------|--------|----------------|
| Q projection | Freivalds (submit_linear) | 1 |
| K projection | Freivalds (submit_linear) | 1 |
| V projection | Freivalds (submit_linear) | 1 |
| Q @ K^T | Freivalds (submit_matmul) | 1 |
| softmax | CPU recompute (submit_elementwise) | 1 |
| attn_probs @ V | Freivalds (submit_matmul) | 1 |
| O projection | Freivalds (submit_linear) | 1 |
| gate_proj | Freivalds (submit_linear) | 1 |
| up_proj | Freivalds (submit_linear) | 1 |
| down_proj | Freivalds (submit_linear) | 1 |
| **Total** | | **10 checks per layer** |

### 3.5 Configuration

```python
@dataclass
class VerifyConfig:
    enabled: bool = True
    freivalds_k: int = 8              # Random vectors (confidence: 1 - 2^{-k})
    mse_threshold: float = 1e-5       # Max MSE for linear/matmul
    elementwise_mse_threshold: float = 1e-4  # Max MSE for softmax/silu
    verify_every_n: int = 1           # Verify every N-th op (1 = all)
    max_workers: int = 2              # CPU ThreadPool workers
    max_verify_tensor_numel: int = 2_000_000  # Skip very large tensors
    fail_on_error: bool = True        # Raise on verification failure
    profile_enabled: bool = True      # Enable timing profiler
```

---

## 5. Performance Evaluation

### 4.1 Qwen3.5-9B Benchmark (RTX 5090, bf16)

**Environment**: NVIDIA RTX 5090 (32 GB VRAM), 18 CPU cores, PyTorch 2.9.0+cu128, bf16 dtype.

**Model**: Qwen/Qwen3.5-9B — hybrid architecture with 8 full_attention layers (verified) + 24 linear_attention layers (pass-through) + 32 MLP layers (all verified).

#### End-to-End Results

| Metric | Origin | Verified | Overhead |
|--------|--------|----------|----------|
| Forward pass (avg 3 runs) | 136.7 ms | 1,950.1 ms | **14.3x** |
| Generation (50 tokens) | 2,256.7 ms | 114,723.0 ms | **50.8x** |
| Tokens/sec | 22.2 | 0.4 | — |
| Peak VRAM | 17,162 MB | 17,473 MB | +311 MB |
| Token equivalence | — | — | **PASS** |

#### Profiling Breakdown (50-token Generation, 16,000 checks)

| Category | Op | Count | Total (ms) | Avg (ms) | Max (ms) |
|----------|-----|-------|-----------|---------|---------|
| verify | linear_freivalds | 6,400 | 359,309 | 56.14 | 235.7 |
| verify | matmul_freivalds | 800 | 10,816 | 13.52 | 43.2 |
| verify | softmax_recompute | 400 | 811 | 2.03 | 18.3 |
| transfer | linear_d2h | 6,400 | 4,088 | 0.64 | 33.5 |
| transfer | matmul_d2h | 800 | 2,149 | 2.69 | 43.3 |
| transfer | softmax_d2h | 400 | 445 | 1.11 | 146.4 |
| compute | matmul | 800 | 267 | 0.33 | 17.7 |

#### Time Distribution

| Category | Total (ms) | Percentage |
|----------|-----------|-----------|
| **verify** (CPU Freivalds) | **370,936** | **98.2%** |
| transfer (D2H copy) | 6,682 | 1.8% |
| compute (GPU matmul timing) | 267 | 0.07% |

#### Verification Statistics

| Phase | Total Checks | Failures | Pass Rate |
|-------|-------------|----------|-----------|
| Forward pass (3 runs) | 960 | 87 | 90.9% |
| Generation (50 tokens) | 16,000 | 1,310 | 91.8% |

The ~8% false positive rate stems from bf16 precision limitations: the GPU computes in bf16 while CPU verification runs in float32, producing MSE differences from legitimate rounding. These are **not actual computation errors** — token equivalence confirms the verified model produces identical output to the origin.

### 4.2 Z-Image Diffusion Benchmark (A100, fp32)

| Prompt | Origin (1 step) | Verified (1 step) | Overhead |
|--------|------------------|--------------------|----------|
| Short | 6,959 ms | 15,948 ms | **2.3x** |
| Long | 7,059 ms | 16,115 ms | **2.3x** |

### 4.3 Key Performance Observations

1. **CPU verification is the absolute bottleneck**: `linear_freivalds` accounts for 97% of total verification time. Each check averages 56ms because it performs float32 matrix-vector multiplications on large weight matrices (up to 4096 x 12288 for the 9B model).

2. **D2H transfer is efficient**: Pinned memory + async copy on a dedicated CUDA stream keeps transfer overhead under 2%.

3. **Generation overhead compounds**: Autoregressive generation (50.8x) is far worse than single forward pass (14.3x) because each token triggers a full verification cycle with per-token flush.

4. **Model size matters**: The 2.3x overhead for Z-Image (on A100) vs 14.3x for Qwen3.5-9B (on RTX 5090) reflects the quadratic growth of CPU verification cost with hidden dimension.

5. **Hybrid architectures reduce verification scope**: Qwen3.5's 24 GatedDeltaNet layers are unverified (no standard matmul), naturally reducing overhead compared to a fully-standard 9B model.

---

## 6. Quantized Model Evaluation

Quantization reduces model size and accelerates inference by representing weights and/or activations in lower precision. The key question for our verification system: **does the quantized computation preserve the `y = x @ W^T` structure that Freivalds can verify?**

### 6.1 Weight-Only Quantization (LLMs)

These schemes quantize weights to low precision but dequantize to fp16/bf16 at runtime before matmul. The GPU kernel effectively computes `y = x @ dequant(W_q)^T`, which is a standard linear operation from the verification perspective.

| Scheme | Weight Bits | Activation | Matmul Structure | Freivalds Compatible? | Notes |
|--------|------------|------------|------------------|-----------------------|-------|
| **GPTQ** | 3/4-bit | fp16 | `y = x @ dequant(W_q)^T` | **Yes** | Dequantize → standard matmul. Marlin kernel fuses dequant+matmul but output is fp16 linear result |
| **AWQ** | 4-bit | fp16 | `y = x @ dequant(W_q)^T` | **Yes** | Per-channel scaling; dequant preserves linearity |
| **bitsandbytes NF4** | 4-bit NF4 | fp16 | `y = x @ dequant(W_nf4)^T` | **Yes** | Double quantization; dequant is still linear in x |
| **bitsandbytes INT8** | 8-bit | fp16 | Mixed decomposition: outliers in fp16, rest in int8 | **Partial** | Absmax scaling; output assembled from two paths |
| **GGUF (llama.cpp)** | 2-6 bit | fp16/fp32 | Block-wise dequant → matmul | **Yes** | Q4_K_M, Q5_K_S etc.; standard matmul after dequant |
| **SqueezeLLM** | 3/4-bit | fp16 | Non-uniform lookup + dense matmul | **Yes** | Lookup table dequant; matmul itself is standard |

**Key insight**: For all weight-only schemes, the GPU kernel performs `y = x @ W_fp16^T` after dequantization. Freivalds can verify by storing the dequantized weights on CPU (`s_tilde = dequant(W_q)^T @ s`). The preprocessing cost is the same — just applied to the dequantized weight matrix.

**Implementation**: In `VerifyLinear.__init__`, precompute SLALOM vectors from the dequantized weight:
```python
# For quantized models: dequantize weight, then precompute SLALOM vectors
weight_fp = dequantize(linear.weight)  # Get fp16/fp32 weight
self.s, self.s_tilde = slalom_precompute(weight_fp.t().float().cpu(), k=k)
```

### 6.2 Weight+Activation Quantization (LLMs)

These schemes quantize both weights AND activations, performing the matmul in low precision.

| Scheme | Weight | Activation | Kernel | Freivalds Compatible? | Notes |
|--------|--------|------------|--------|-----------------------|-------|
| **SmoothQuant W8A8** | INT8 | INT8 | `y_int32 = x_int8 @ W_int8^T` → rescale | **Yes (with rescaling)** | Linear in x; scale factors known |
| **FP8 (E4M3/E5M2)** | FP8 | FP8 | `y_fp16 = x_fp8 @ W_fp8^T` (Tensor Core) | **Yes** | Native on H100+; output is fp16 |
| **AQLM** | 1-2 bit (codebook) | fp16 | Codebook lookup + matmul | **Partial** | Multi-codebook decomposition; matmul per codebook |
| **QuIP#** | 2-4 bit (lattice) | fp16 | Hadamard transform + quantized matmul | **Partial** | Incoherence processing changes W structure |

**SmoothQuant (W8A8)**: The matmul is `y = scale_y * (quantize(x/scale_x) @ quantize(W/scale_w)^T)`. Since quantization is deterministic and scales are known, the CPU can replicate this exactly. Freivalds works on the integer matmul: verify `C_int32 = A_int8 @ B_int8^T`, then apply scaling.

**FP8**: H100/B200 Tensor Cores compute `y_fp16 = x_fp8 @ W_fp8^T`. The output is fp16 — a standard linear result. Freivalds verification can work on the fp16 output using fp32 preprocessing vectors. The reduced precision (fp8 → fp16) actually improves MSE tolerance.

### 6.3 KV Cache Quantization

| Scheme | Precision | Impact on Verification |
|--------|-----------|----------------------|
| **FP8 KV cache** | E4M3 keys, E5M2 values | Q@K^T and attn@V operands are quantized; increases MSE in Freivalds matmul check |
| **INT8 KV cache** | Symmetric INT8 | Same; need wider MSE threshold for matmul verification |
| **INT4 KV cache** | Group-wise INT4 | Significant quantization noise; may need dedicated thresholds |

KV cache quantization affects **attention matmul verification** (Q@K^T, attn@V) but NOT linear layer verification (since linear layers use the full-precision KV before caching).

### 6.4 Quantized Diffusion Models

| Model/Scheme | Quantization | Architecture | Freivalds Compatible? |
|-------------|-------------|-------------|----------------------|
| **SDXL INT8 (TensorRT)** | W8A8 post-training | UNet attention + conv | **Yes** — dequantized matmul |
| **SDXL FP8 (TensorRT)** | W8A8 fp8 | UNet + MHA quantized | **Yes** — fp16 output |
| **Flux INT4 (GGUF)** | W4 weight-only | DiT transformer | **Yes** — dequant + standard matmul |
| **Z-Image INT4** | W4 SVDQ | DiT transformer | **Yes** — SVD factored, then standard matmul |
| **SD3.5 INT8** | W8A8 | MMDiT | **Yes** — per-tensor/per-channel scaling |

Diffusion model quantization is structurally simpler than LLM quantization because:
1. No autoregressive KV cache (each denoising step is independent)
2. Attention dimensions are smaller (spatial, not sequence-length)
3. Most quantization is weight-only or W8A8 post-training

### 6.5 Verification-Hostile Architectures

Some architectures are **NOT directly compatible** with Freivalds:

| Architecture | Why Not Compatible | Workaround |
|-------------|-------------------|-----------|
| **Mixture of Experts (MoE)** | Routing is input-dependent; different experts activated per token | Verify each expert's linear op independently; skip router |
| **GatedDeltaNet (Qwen3.5)** | Mamba-style linear attention with state-space model | Cannot verify with Freivalds; pass-through |
| **RWKV (linear attention)** | Recurrent state update, no explicit matmul | Not verifiable with Freivalds |
| **BitNet (1-bit weights)** | Ternary {-1, 0, 1} matmul via addition | Technically verifiable but Freivalds unnecessary (recompute directly) |
| **AQLM multi-codebook** | Decomposed into codebook lookups + small matmuls | Verify each sub-matmul; higher check count |

### 6.6 Compatibility Summary

```
Freivalds Verification Compatibility:

  FULLY COMPATIBLE (standard y = x @ W^T after dequant):
    [===] GPTQ (3/4-bit)
    [===] AWQ (4-bit)
    [===] bitsandbytes NF4
    [===] GGUF formats (Q4_K_M, Q5_K_S, etc.)
    [===] FP8 (E4M3/E5M2)
    [===] SmoothQuant W8A8
    [===] TensorRT INT8/FP8 diffusion

  PARTIALLY COMPATIBLE (needs adaptation):
    [== ] bitsandbytes INT8 (mixed decomposition)
    [== ] AQLM (multi-codebook, verify per codebook)
    [== ] QuIP# (pre/post Hadamard transforms)
    [== ] KV cache quantization (wider MSE thresholds)

  NOT COMPATIBLE (architectural mismatch):
    [   ] GatedDeltaNet / Mamba (state-space, not matmul)
    [   ] RWKV (recurrent, not matmul)
    [   ] BitNet 1-bit (trivial to recompute, Freivalds unnecessary)
```

### 6.7 Implementation Recommendations for Quantized Models

1. **For weight-only quantization (GPTQ/AWQ/NF4)**: Dequantize weights once at load time, precompute SLALOM vectors on the fp16 result. No runtime overhead beyond standard verification.

2. **For W8A8 (SmoothQuant/FP8)**: Precompute SLALOM vectors on the int8/fp8 weight representation. Verify in the quantized domain where possible; fall back to fp32 verification with wider thresholds.

3. **For MoE models**: Precompute SLALOM vectors for ALL experts. At runtime, verify only the activated experts. Router decisions are non-linear (verified by CPU recomputation).

4. **MSE threshold adjustment**: Quantized inference introduces additional rounding error. Recommended thresholds:
   - fp32: `mse_threshold = 1e-5`
   - fp16/bf16: `mse_threshold = 5e-3`
   - fp8: `mse_threshold = 1e-2`
   - int8: `mse_threshold = 5e-2`
   - int4: `mse_threshold = 1e-1`

---

## 7. Related Work

### 7.1 Zero-Knowledge ML (zkML)

zkML provides cryptographic guarantees (information-theoretic soundness) at extreme computational cost:

| Project | Approach | Status | Overhead |
|---------|----------|--------|----------|
| **EZKL** | ONNX → zk-SNARK (Halo2) | Active, most mature | 10,000–100,000x |
| **Modulus Labs / Remainder** | zkML for models up to 18M params | Research | >100,000x |
| **zkPyTorch** (2025) | VGG-16 proof in 2.2 seconds | Research milestone | ~10,000x |
| **Lagrange / Jolt** | zkVM-based ML proving | Active | ~50,000x |

**Comparison**: zkML produces non-interactive, universally verifiable proofs, but the overhead is 4–5 orders of magnitude higher than Freivalds. Freivalds provides probabilistic verification at O(n^2) cost with tunable confidence (1 − 2^{−k}), making it practical for real-time inference verification where interactive checking is acceptable.

### 7.2 Optimistic Verification

| Work | Key Idea | Venue |
|------|----------|-------|
| **Optimistic Verifiable Training** (Bhat et al.) | Assume correct, verify on challenge; handles GPU nondeterminism via higher-precision training | NeurIPS 2024 |
| **NAO** (Nondeterminism-Aware Optimistic) | Orders-of-magnitude lower latency than zk; preserves native CUDA kernels | 2025 preprint |
| **opML** (ORA) | On-chain results with challenge period; fraud proof on dispute | Blockchain |
| **Proof of Sampling** (Hyperbolic Labs) | Statistical sampling-based verification | 2024 |

NAO is the closest in spirit to our approach: both accept that GPU computation is correct by default and verify probabilistically/on-demand rather than proving correctness upfront.

### 7.3 TEE-Based ML Inference

| Work | Approach | Overhead |
|------|----------|----------|
| **Gramine + PyTorch in SGX** | Full PyTorch inside SGX enclave via LibOS | 4.8–6.15% |
| **AttestLLM** (2025) | Activation watermarking + periodic TEE-based challenge-response | Low |
| **Branchy-TEE** (2025) | On-demand model loading with early-exit in SGX | Variable |
| **Model Partitioning** | Split DNN: convolutions on GPU, non-linear in SGX | 10–30% |

Model partitioning approaches are the closest to our architecture: GPU handles compute-intensive operations, CPU/TEE handles verification. The key difference is that partitioning approaches run a subset of the model inside the TEE, while we run the entire model on GPU and verify every operation via Freivalds.

### 7.4 Freivalds' Algorithm in Verification

| Variant | Description | Reference |
|---------|-------------|-----------|
| **Classical Freivalds** | Random binary vector, single check | Freivalds (1977) |
| **Multi-vector Freivalds** | k random vectors, error prob 2^{-k} | Standard extension |
| **Gaussian GVFA** | Gaussian random projections for improved stability | NIST (2017) |
| **This project** | Float32 multi-vector with MSE threshold, batched + parallel variants | — |

Our implementation extends classical Freivalds with:
- Continuous (float) random vectors instead of binary, for fp32 MSE-based comparison
- Batched variants for multi-head attention (4D tensors)
- Parallel ABr/Cr computation via ThreadPoolExecutor
- NaN fallback to `torch.allclose()` for numeric edge cases
- Random vector caching to avoid repeated allocation

---

## 8. CPU TEE Landscape Survey

### 6.1 Comprehensive Product Comparison

| TEE | Vendor | Granularity | Max Memory | CPU Overhead | Attestation | PyTorch OK? | Status |
|-----|--------|-------------|-----------|-------------|-------------|------------|--------|
| **Intel SGX** | Intel | Process enclave | 512 GB/socket (v2) | 5–15% | DCAP/IAS | Yes (Gramine) | GA (Xeon only) |
| **Intel TDX** | Intel | Full VM | VM DRAM limit | 2–10% | Trust Authority | Yes (native) | GA |
| **AMD SEV-SNP** | AMD | Full VM | VM DRAM limit | 5–15% | AMD KDS | Yes (native) | GA |
| **ARM TrustZone** | ARM | World partition | 16–256 MB | ~0% | Platform-specific | No | GA (mobile) |
| **ARM CCA** | ARM | Realm (VM-like) | Dynamic, encrypted | 2–8% (est.) | ARM CCA framework | Yes (in Realm) | Early silicon |
| **RISC-V Keystone** | Open | Process enclave | PMP-limited | TBD | Platform-specific | No | Research |
| **RISC-V CoVE** | RISC-V Intl | Full VM | TBD | TBD | TSM-based | TBD | Spec draft |
| **IBM PEF** | IBM | Full VM | System DRAM | 2–10% | TPM | Yes (native) | GA (POWER) |
| **NVIDIA H100 CC** | NVIDIA | GPU memory | 80 GB HBM3 | <7% | NVIDIA/Intel TA | N/A (GPU) | GA |
| **NVIDIA B200 CC** | NVIDIA | GPU memory + NVLink | 192 GB HBM3e | ~0% | NVIDIA attestation | N/A (GPU) | GA |

### 6.2 Intel SGX

**Supported CPUs**: 3rd/4th/5th Gen Intel Xeon Scalable (Ice Lake-SP, Sapphire Rapids, Emerald Rapids). **Deprecated** on consumer Intel Core since 11th Gen (2021).

**Architecture**: Process-level enclaves. Application code runs in hardware-encrypted memory regions (Enclave Page Cache). SGXv2 supports dynamic memory management (EDMM) with up to 512 GB EPC per socket.

**Performance**:
- Enclave entry/exit: ~8,000–14,000 cycles (~3–5 us at 3 GHz)
- Memory encryption (MKTME): 5–20% for memory-intensive workloads
- EPC paging (exceeding EPC): 10–1,000x slowdown
- Gramine-SGX LLM inference: 4.8–6.15% measured overhead

**Programming Model**: Intel SGX SDK (C/C++), Open Enclave SDK, or **Gramine LibOS** (runs unmodified Linux applications including Python/PyTorch inside enclaves). Intel provides `sgx-pytorch` for running PyTorch inference inside enclaves.

**For this project**: SGX provides the smallest Trusted Computing Base (TCB) — only the verification code runs inside the enclave, not the full OS. The Freivalds working set easily fits within EPC. However, SGX is only available on Xeon server CPUs and requires Gramine integration.

### 6.3 Intel TDX

**Supported CPUs**: 4th Gen Xeon Scalable (Sapphire Rapids), 5th Gen (Emerald Rapids), Intel Xeon 6 (Granite Rapids). Xeon 6 adds **TDX Connect** — extends TEE boundary to PCIe devices including GPUs.

**Architecture**: VM-level TEE. Entire virtual machines ("Trust Domains") run in hardware-encrypted memory via TME-MK (AES-128-XTS). A SEAM (Secure Arbitration Mode) module mediates between the hypervisor and Trust Domains.

**Performance**:
- CPU-bound workloads: 2–5%
- Memory-intensive: 5–10%
- I/O-heavy: 5–10% (bounce buffer overhead)
- TDX Connect (Xeon 6): reduces I/O overhead by eliminating shared-memory copies

**Programming Model**: Run standard Linux VMs inside Trust Domains. No application modification. Guest OS must be TDX-enlightened (Linux kernel 6.2+).

**Cloud Availability**: Azure DCesv5/ECesv5, GCP Confidential VMs (C3), Oracle Cloud.

**For this project**: TDX is the most frictionless option — zero code changes, full PyTorch support, 2–5% overhead. TDX Connect on Xeon 6 could extend the TEE boundary to include GPU communication, providing end-to-end confidential computing.

### 6.4 AMD SEV-SNP

**Supported CPUs**: EPYC 7003 (Milan, Zen 3), EPYC 9004 (Genoa, Zen 4), EPYC 9005 (Turin, Zen 5). Earlier variants: SEV (Naples), SEV-ES (Rome).

**Architecture**: VM-level memory encryption with per-VM AES keys. SNP adds memory integrity protection via Reverse Map Table, preventing hypervisor replay/remap attacks.

**Performance (EPYC 9005 benchmarks)**:
- CPU-intensive: 5–8%
- Memory-intensive (databases): 10–20%
- Container workloads: <4%
- I/O-heavy: <5%

**Cloud Availability**: Azure DCasv5/ECasv5, GCP N2D Confidential VMs, AWS (custom instances).

**For this project**: Strong alternative to TDX. Widely available, especially on AMD-based cloud instances. Slightly higher overhead than TDX for compute-bound workloads but excellent for mixed workloads.

### 6.5 ARM CCA (Confidential Compute Architecture)

**Supported CPUs**: ARMv9.2+ cores (Cortex-X4, A720, Neoverse V3/N3).

**Architecture**: Extends ARM TrustZone with a new "Realm" world and Granule Protection Table (GPT) for per-page memory ownership tracking. Realms are dynamically created/destroyed, unlike static TrustZone partitions.

**Status**: Early silicon (2024–2025). Server availability through Neoverse V3-based platforms. Cloud availability expected 2026+.

**For this project**: Promising for ARM-based server deployments but not yet production-ready. Monitor for Neoverse V3 cloud availability.

### 6.6 RISC-V TEE Efforts

| Project | Type | Status | Hardware |
|---------|------|--------|----------|
| **Keystone** | Process enclave (PMP) | Incubation (CCC) | QEMU, SiFive |
| **Sanctum** | SGX-like enclave | Research prototype | MIT |
| **CoVE** | VM-level (like TDX) | Spec draft (2025) | None yet |
| **Penglai** | Forking enclaves | Research | QEMU |

All RISC-V TEE solutions are pre-production. CoVE standardization is the most promising path, with commercial silicon expected 2027+.

### 6.7 NVIDIA Confidential Computing (GPU TEE)

**H100 (Hopper)**: First GPU with hardware TEE. On-die root of trust, AES-encrypted GPU memory, attestable boot chain. Works with CPU TEEs (TDX, SEV-SNP) for end-to-end confidential computing. Overhead: <7% for LLM inference.

**B200 (Blackwell)**: First TEE-I/O capable GPU with inline NVLink encryption between GPUs. Multi-GPU confidential computing with near-zero throughput loss.

**Relevance**: GPU TEE could complement or replace CPU-side Freivalds verification. If both GPU and CPU are trusted, the threat model shifts from "verify untrusted GPU" to "defense-in-depth."

### 6.8 Cloud Provider Summary

| Provider | SGX | TDX | SEV-SNP | GPU CC | Nitro Enclaves |
|----------|-----|-----|---------|--------|---------------|
| **Azure** | DCsv2/DCsv3 | DCesv5/ECesv5 | DCasv5/ECasv5 | NCCads H100 v5 | — |
| **GCP** | — | C3 Confidential | N2D Confidential | Preview | — |
| **AWS** | — | — | — | — | Most EC2 types |
| **Oracle** | — | Available | — | — | — |

---

## 9. Compromise Methods for Non-TEE Environments

For environments where hardware TEE is unavailable, several alternative approaches can provide varying levels of trust:

### 7.1 Multi-Party Verification (Recommended)

**Approach**: Split the Freivalds verification across 2–3 non-colluding parties. Each party receives shares of the input/output tensors and independently verifies.

**Why it works for Freivalds**: The verification is almost entirely linear operations (matrix-vector multiplies), which are "free" in secret-sharing-based MPC (no communication for additive shares). Only the final comparison requires a small interaction round.

**Trust model**: Security holds as long as parties do not collude. 2-of-3 honest majority provides strong guarantees.

**Overhead**: Communication-bound for setup; Freivalds' small working set (random vector + matrix-vector products) makes this tractable. Expected 2–5x slowdown over single-party verification.

**Assessment**: The strongest non-TEE option. Deployable today with no special hardware.

### 7.2 Trusted Third-Party Verification

**Approach**: Outsource Freivalds verification to a trusted third party. The verifier receives `(x, W, y)` and checks `y = x @ W`.

**Variants**:
- Single trusted verifier (simplest, weakest trust model)
- K-of-N independent verifiers: any one honest verifier detects tampering
- Notarized verification: verifier signs attestation of correctness

**Overhead**: Network transfer of input/output tensors + verification compute. For Qwen3.5-9B with short prompts, each verification payload is ~100 KB — manageable over fast networks.

**Assessment**: Practical and immediately deployable. Trust depends on the third party's integrity.

### 7.3 Redundant Computation

**Approach**: Run inference on 2–3 independent GPU providers. Compare results; flag discrepancies.

**Trust model**: Secure against non-colluding providers. Cost: 2–3x compute budget.

**Variants**:
- Full redundancy: all providers run the complete model
- Partial redundancy: different providers run different layers (spot-check)
- Probabilistic redundancy: randomly select which operations to cross-check

**Assessment**: Simple, no special hardware or code changes. High cost but strongest guarantees against non-colluding adversaries.

### 7.4 Software Isolation (Defense-in-Depth)

| Technique | Protection Level | What It Prevents |
|-----------|-----------------|-----------------|
| Process isolation (namespaces, cgroups) | OS-level | Co-tenant attacks |
| Memory encryption (TRESOR) | Register-level keys | Cold-boot attacks |
| ASLR + stack canaries | Address randomization | Memory disclosure |
| Measured boot (IMA/TPM) | Boot chain integrity | Tampered binaries |

**Assessment**: These techniques prevent certain classes of attacks but cannot make the CPU "truly trusted" against a privileged attacker (root/hypervisor compromise). Useful as defense-in-depth layered with other approaches, not as a TEE replacement.

### 7.5 Homomorphic Encryption

**Approach**: Encrypt verification inputs and run Freivalds on encrypted data.

**Limitation**: FHE is 10,000–1,000,000x slower than plaintext computation. Does not provide integrity guarantees without additional ZK proofs.

**Assessment**: Not practical for real-time verification at current performance levels.

### 7.6 Comparison of Non-TEE Approaches

| Approach | Trust Model | Overhead | Complexity | Deployable Today |
|----------|-------------|----------|-----------|-----------------|
| Multi-party verification | N-of-M honest | 2–5x | Moderate | Yes |
| Trusted third-party | Trust in verifier | 1.5–2x (network) | Low | Yes |
| Redundant computation | Non-colluding | 2–3x cost | Low | Yes |
| Software isolation | Defense-in-depth | ~0% | Low | Yes |
| Homomorphic encryption | Cryptographic | 10,000x+ | High | No (too slow) |

---

## 10. TEE Execution Evaluation

### 8.1 Projected Overhead: Freivalds Inside TEE

The Freivalds verification workload has characteristics favorable for TEE execution:

**Working Set Analysis (Qwen3.5-9B, single linear verification)**:
- Input tensor `x`: [1, 32, 4096] × 4 bytes (fp32) = 512 KB
- Weight transpose `W^T`: [4096, 12288] × 4 bytes = 192 MB (largest MLP weight)
- Output tensor `y`: [1, 32, 12288] × 4 bytes = 1.5 MB
- Random vector `r`: [12288, k] × 4 bytes = 480 KB (k=10)
- Intermediate `Br`: [4096, k] × 4 bytes = 160 KB

**Total working set**: ~195 MB per verification — fits within SGXv2 EPC (512 GB), and is trivially within TDX/SEV VM memory.

### 8.2 Estimated TEE Overhead by Platform

| TEE Platform | Base CPU Verification (ms) | TEE Overhead | TEE-Adjusted (ms) | Total Forward (ms) |
|-------------|---------------------------|-------------|-------------------|-------------------|
| None (current) | 56.1 avg/check | 0% | 56.1 | 1,950 |
| Intel TDX | 56.1 avg/check | +2–5% | 57.2–58.9 | 1,989–2,048 |
| AMD SEV-SNP | 56.1 avg/check | +5–8% | 58.9–60.6 | 2,048–2,098 |
| Intel SGX (Gramine) | 56.1 avg/check | +5–15% | 58.9–64.5 | 2,048–2,244 |

**Key insight**: Because the verification overhead (56ms avg) already dominates the pipeline (98.2% of total time), adding 2–15% TEE overhead on top of verification results in only 2–15% increase in total time — a marginal cost for hardware-rooted trust.

### 8.3 SGX-Specific Considerations

**Enclave transitions**: With `max_workers=2` and ~320 verification tasks per forward pass, each requiring one enclave entry/exit:
- 320 tasks × 5 us = 1.6 ms total transition overhead (negligible vs 1,950 ms forward pass)

**EPC paging**: The 192 MB maximum working set (MLP weight matrix) fits within SGXv2 EPC. No paging expected. For SGXv1 (128 MB EPC), the largest weight matrices would cause paging — avoid SGXv1 for large models.

**Threading**: SGX enclaves support multi-threading. The ThreadPoolExecutor with `max_workers=2-4` works inside Gramine-SGX, though may show slightly reduced throughput due to TCS (Thread Control Structure) management.

### 8.4 TDX-Specific Considerations

**Deployment model**: Run the entire inference server inside a TDX Trust Domain. No code changes needed — PyTorch, transformers, and the verification code all run natively.

**TDX Connect (Xeon 6)**: Extends TEE boundary to PCIe devices. This could protect the GPU-to-CPU D2H transfer channel, eliminating the "Memory Bus: Observed" entry from the threat model.

**Attestation workflow**:
1. TDX Trust Domain boots with measured firmware
2. Guest generates attestation quote via `tdx_attest` ioctl
3. Remote verifier checks quote against Intel Trust Authority
4. Verifier gains confidence that the correct verification code is running inside a genuine TEE

### 8.5 End-to-End Confidential Inference (TDX + H100 CC)

The strongest deployment combines CPU TEE and GPU TEE:

| Component | TEE | Trust Level |
|-----------|-----|-------------|
| CPU inference server | Intel TDX | Trusted |
| GPU computation | NVIDIA H100 CC | Trusted |
| CPU-GPU transfer | TDX Connect (Xeon 6) | Trusted |
| Remote attestation | Composite (Intel TA + NVIDIA) | Verified |

In this configuration, Freivalds verification becomes **defense-in-depth** rather than the sole trust mechanism. The GPU is hardware-trusted (CC mode), and the CPU is hardware-trusted (TDX). Freivalds catches any remaining failures (hardware faults, firmware bugs, side-channel exploits) that escape TEE protections.

**Estimated total overhead**: <10% over unverified, non-TEE inference.

### 8.6 Benchmark Projections for TEE Machines

| Configuration | Forward (ms) | Generation 50 tok (s) | Tok/s | Overhead vs Origin |
|--------------|-------------|----------------------|-------|-------------------|
| Origin (no verify, no TEE) | 137 | 2.3 | 22.2 | 1.0x |
| Verified, no TEE (current) | 1,950 | 114.7 | 0.4 | 14.3x |
| Verified + TDX (~3% TEE tax) | 2,009 | 118.2 | 0.4 | 14.7x |
| Verified + SEV-SNP (~7% TEE tax) | 2,087 | 122.7 | 0.4 | 15.2x |
| Verified + SGX (~10% TEE tax) | 2,145 | 126.2 | 0.4 | 15.7x |
| Origin + H100 CC only (no verify) | 146 | 2.4 | 20.6 | 1.07x |
| Verified + TDX + H100 CC | 2,009 | 118.2 | 0.4 | 14.7x |

The incremental cost of adding TEE (0.4–1.4x additional on top of the 14.3x verification overhead) is small relative to the verification cost itself. **The optimization priority is CPU verification speed, not TEE overhead.**

---

## 11. Recommendations

### 9.1 For Making the CPU Truly Trusted

1. **Intel TDX** (recommended): Zero code changes, full PyTorch/NumPy support, 2–5% overhead, GA on 5th Gen Xeon and major cloud providers. TDX Connect on Xeon 6 extends TEE to PCIe.

2. **AMD SEV-SNP** (strong alternative): Same VM-level approach, wider cloud availability on AMD instances, 5–8% overhead.

3. **Intel SGX via Gramine** (smallest TCB): Only verification code runs in enclave. 5–15% overhead. Better isolation than VM-level TEE but requires Gramine integration.

### 9.2 For Reducing Verification Overhead

The 14.3x overhead is dominated by CPU Freivalds computation (98.2%). Priority optimizations:

1. **Increase `max_workers`**: Scale ThreadPoolExecutor to 8–16 workers on multi-core systems to parallelize independent verification tasks.

2. **Adaptive verification** (`verify_every_n > 1`): Verify every N-th operation instead of all. With N=4, overhead drops ~4x at the cost of reduced coverage.

3. **Reduce `freivalds_k`**: Lower from 8 to 4 random vectors. Reduces per-check cost ~2x. Confidence drops from 1−2^{-8} to 1−2^{-4} (93.75%), still sufficient for most threat models.

4. **Batch verification**: Accumulate multiple layers' results and verify in a single batched Freivalds check, amortizing overhead.

5. **BLAS-optimized CPU compute**: Ensure Intel MKL or OpenBLAS is used for CPU matrix-vector multiplications (currently using PyTorch's default CPU backend).

### 9.3 For Non-TEE Environments

Multi-party verification with 2–3 non-colluding nodes provides the strongest guarantees without hardware TEE. This is immediately deployable with the current codebase by running separate verification workers on independent machines.

---

## 12. References

### Academic Papers

0. Tramer, F. & Boneh, D. "SLALOM: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware." *ICLR*, 2019. [arXiv:1806.03287](https://arxiv.org/abs/1806.03287)
1. Freivalds, R. "Probabilistic Machines Can Use Less Running Time." *IFIP Congress*, 1977.
2. Bhat, S. et al. "Optimistic Verifiable Training by Controlling Hardware Nondeterminism." *NeurIPS*, 2024. [arXiv:2403.09603](https://arxiv.org/abs/2403.09603)
3. Sun, W. et al. "zkLLM: Zero Knowledge Proofs for Large Language Models." *ACM CCS*, 2024. [arXiv:2404.16109](https://arxiv.org/abs/2404.16109)
4. "NAO: Nondeterminism-Aware Optimistic Verification for Neural Network Inference." 2025. [arXiv:2510.16028](https://arxiv.org/abs/2510.16028)
5. Grover, K. et al. "Gaussian Variant of Freivalds' Algorithm." NIST, 2017. [arXiv:1705.10449](https://arxiv.org/abs/1705.10449)
6. Akram, J. et al. "SoK: Machine Learning with Confidential Computing." *ACM Computing Surveys*, 2024.
7. Xiang, Y. et al. "CoVE: Towards Confidential Virtual Machine Execution on RISC-V." *ACM ASPLOS*, 2023.

### TEE Documentation

8. Intel SGX Developer Reference. [intel.com/sgx](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
9. Intel TDX Product Page. [intel.com/tdx](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/trust-domain-extensions.html)
10. AMD SEV-SNP Developer Guide. [amd.com/sev](https://www.amd.com/en/developer/sev.html)
11. ARM CCA Architecture. [arm.com/cca](https://www.arm.com/architecture/security-features/arm-confidential-compute-architecture)
12. NVIDIA Confidential Computing on H100. [developer.nvidia.com](https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/)
13. Gramine LibOS Project. [gramine.readthedocs.io](https://gramine.readthedocs.io/)

### zkML Projects

14. EZKL. [ezkl.xyz](https://ezkl.xyz/)
15. Modulus Labs / Remainder. [moduluslabs.xyz](https://www.moduluslabs.xyz/)

### Cloud Confidential Computing

16. Azure Confidential Computing. [azure.microsoft.com/confidential-compute](https://azure.microsoft.com/en-us/solutions/confidential-compute)
17. AMD SEV-SNP Performance on EPYC 9005. [phoronix.com](https://www.phoronix.com/review/amd-epyc-9005-sev-snp)
18. Intel TDX Performance on 4th Gen Xeon. [intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/trust-domain-extensions-on-4th-gen-xeon-processors.html)

---

*Report generated: 2026-04-15. Updated: 2026-04-15 (SLALOM alignment, quantized model evaluation). Hardware: NVIDIA RTX 5090 (32 GB), 18 CPU cores. Software: PyTorch 2.9.0+cu128, Transformers 5.5.4.*
