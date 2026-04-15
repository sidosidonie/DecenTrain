# Design Document: Verified Neural Network Inference

## 1. Overview

This project implements **probabilistic verification of GPU-computed neural network inference** using Freivalds' algorithm. The system targets a threat model where the GPU is untrusted (potentially returning incorrect results), while the CPU operates within a Trusted Execution Environment (TEE). Verification runs asynchronously on CPU, overlapping with GPU forward passes to minimize latency overhead.

Two model families are supported:
- **Verified LLM**: Llama-family models (attention + MLP layers)
- **Verified Diffusers**: Z-Image diffusion transformer (attention + feedforward layers), plus MLIR compilation tooling

---

## 2. Threat Model & Motivation

| Component | Trust Level | Role |
|-----------|-------------|------|
| CPU (TEE) | Trusted | Verification, non-matmul ops |
| GPU | Untrusted | Accelerated matmul/linear ops |
| Memory Bus | Observed | GPU-to-CPU async transfer |

**Assumption**: The GPU may be compromised, faulty, or performing incorrect computation. Only `matmul` and `linear` operations are offloaded to GPU; all other operations (activations, normalization, element-wise) execute on trusted CPU. Every GPU-computed result is verified before being accepted.

---

## 3. Core Algorithm: Freivalds' Verification

### 3.1 Principle

To verify `C = A @ B + bias` without full recomputation:

1. Generate random vector `r` of shape `(p, k)`, where `k` controls confidence
2. Compute `Br = B @ r` and `ABr = A @ Br` on CPU
3. Compute `Cr = C @ r` on CPU
4. Check: `MSE(ABr + bias @ r, Cr) < threshold`

**Complexity**: O(n^2 * k) vs O(n^3) for full recomputation.  
**Confidence**: With `k=10` iterations, error probability < 2^{-10} ~ 0.1%.

### 3.2 Variants Implemented

| Function | Location | Description |
|----------|----------|-------------|
| `freivalds_algorithm_2d_bias` | `verified_llm/verify_linear.py` | Standard 2D matmul with bias |
| `freivalds_batch_matmul_bias` | `verified_llm/verify_linear.py` | Batched matmul with bias |
| `freivalds_batch_matmul_parallel` | `verified_llm/verify_linear.py` | Parallel CPU/GPU via ThreadPool |
| `freivalds_algorithm_stream` | `verified_llm/verify_linear.py` | Stream-based with event sync |

### 3.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 10 | Number of random vectors (confidence level) |
| `mse_threshold` | 1e-5 | Maximum acceptable MSE |
| `max_verify_tensor_numel` | 2,000,000 | Skip verification for tensors above this size |

---

## 4. System Architecture

```
+--------------------------------------------------------------------+
|                     User-Facing Pipeline API                        |
|  (VerifiedZImagePipeline / create_llm_model)                       |
+--------------------------------------------------------------------+
          |                                        |
          v                                        v
+---------------------------+      +---------------------------+
| Verified Diffusers Stack  |      |   Verified LLM Stack      |
| (Z-Image Transformer)     |      |   (Llama Transformer)     |
+---------------------------+      +---------------------------+
| VerifiedZImageTransformer |      | LlamaAttentionVerify      |
|   TransformerBlock (x N)  |      | LlamaMLPVerify            |
|     Attention + FFN       |      | replace_attn/replace_mlp  |
+---------------------------+      +---------------------------+
          |                                        |
          v                                        v
+--------------------------------------------------------------------+
|                   Verification Runtime Layer                        |
|  VerifyRuntime / ChunkedVerifyRuntime                               |
|  - compute_stream (GPU forward pass)                                |
|  - copy_stream (async GPU->CPU pinned memory)                       |
|  - ThreadPoolExecutor (CPU Freivalds verification)                  |
|  - Event synchronization                                            |
+--------------------------------------------------------------------+
          |
          v
+--------------------------------------------------------------------+
|                   Freivalds Verification Core                       |
|  verify_linear.py: freivalds_algorithm_2d_bias, batch variants      |
+--------------------------------------------------------------------+
          |
          v
+--------------------------------------------------------------------+
|                   Profiling & Observability                         |
|  VerifyProfiler -> CSV / matplotlib plots                           |
+--------------------------------------------------------------------+
```

---

## 5. Async Pipelined Execution Model

The key performance optimization is **overlapping GPU computation with CPU verification** using three concurrent execution lanes:

```
Time --->

GPU (compute_stream):
  [Linear_1] [Linear_2] [Linear_3] [Linear_4] ...
       |          |          |          |
       v          v          v          v
Copy (copy_stream, pinned memory):
     [Copy_1]  [Copy_2]  [Copy_3]  [Copy_4] ...
        |          |          |          |
        v          v          v          v
CPU (ThreadPool):
      [Verify_1] [Verify_2] [Verify_3] [Verify_4] ...
```

### 5.1 Synchronization

- **CUDA Events**: `copy_stream` waits on `compute_stream` via `wait_event()` before starting D2H transfer
- **Pinned Memory**: Pre-allocated page-locked CPU buffers for maximum PCIe bandwidth
- **Thread Pool**: `ThreadPoolExecutor(max_workers=2)` runs Freivalds checks without blocking GPU
- **Flush barrier**: `runtime.flush()` blocks until all pending verifications complete (used at layer boundaries or end of inference)

### 5.2 Chunked Runtime (Large Tensors)

For tensors exceeding pinned memory budgets, `ChunkedVerifyRuntime` breaks GPU-to-CPU copies into row-wise chunks, avoiding allocation failures while maintaining the same verification guarantees.

---

## 6. Module Design

### 6.1 Verified LLM (`verified_llm/`)

#### Layer Injection

```python
# Recursive module replacement pattern
def replace_attn(model):
    for name, child in model.named_children():
        if isinstance(child, LlamaAttention):
            setattr(model, name, LlamaAttentionVerify(child))
        else:
            replace_attn(child)
```

#### Attention Layer (`attn_layer.py`)

`LlamaAttentionVerify` wraps the native Llama attention with verification:

- Q, K, V projections: Each verified via `VerifyLinear`
- Q @ K^T matmul: Verified via `freivalds_batch_matmul_bias`
- Softmax + Dropout: Computed on CPU (not a matmul)
- Output projection: Verified via `VerifyLinear`
- Optional noise injection for robustness testing

#### MLP Layer (`mlp_layer.py`)

`LlamaMLPVerify` wraps the SwiGLU MLP with pipelined verification:

```
Stage 1: gate_proj(x)
Stage 2: gate_proj_verify || up_proj(x)
Stage 3: up_proj_verify || SiLU(gate) * up -> down_proj
Stage 4: down_proj_verify
```

### 6.2 Verified Diffusers (`verified_diffusers/`)

#### Z-Image Module Hierarchy

```
VerifiedZImagePipeline
  └── VerifiedZImageTransformer2DModel
        ├── noise_refiner blocks (patched)
        ├── context_refiner blocks (patched)
        ├── main transformer layers (patched)
        │     └── VerifiedZImageTransformerBlock (x N)
        │           ├── VerifiedZImageAttention
        │           │     ├── VerifyLinearModule (to_q, to_k, to_v, to_out)
        │           │     └── VerifyMatmul (Q@K^T, attn@V)
        │           └── VerifiedZImageFeedForward
        │                 └── VerifyLinearModule (w1, w2, w3)
        └── embeddings (x_embedder, t_embedder, cap_embedder)
```

#### Layer Wrappers (`zimage/layers.py`)

| Class | Wraps | Verification |
|-------|-------|--------------|
| `VerifyLinearModule` | `nn.Linear` | Stores W^T and bias on CPU (pinned); submits Freivalds check after GPU forward |
| `VerifyMatmul` | `torch.matmul` | Submits batch matmul verification |

#### Runtime (`zimage/runtime.py`)

```python
class VerifyRuntime:
    compute_stream: torch.cuda.Stream   # GPU forward pass
    copy_stream: torch.cuda.Stream      # Async GPU->CPU transfer
    _executor: ThreadPoolExecutor       # CPU verification workers
    _futures: list[Future]              # Pending verification tasks
    config: VerifyConfig                # Thresholds, k, toggle
    profiler: VerifyProfiler            # Timing collection
```

Key methods:
- `submit_linear(tag, x, weight_t, bias, y)` - Queue linear verification
- `submit_matmul(tag, a, b, c)` - Queue matmul verification
- `flush()` - Wait for all pending verifications
- `shutdown()` - Cleanup

#### Configuration (`zimage/config.py`)

```python
@dataclass
class VerifyConfig:
    enabled: bool = True
    freivalds_k: int = 8
    mse_threshold: float = 1e-5
    verify_every_n: int = 1       # Verify every Nth op (for adaptive)
    max_workers: int = 2
    fail_on_error: bool = True
    profile_enabled: bool = True
    max_verify_tensor_numel: int = 2_000_000
```

### 6.3 MLIR Compilation (`verified_diffusers/compile*.py`)

Separate from the runtime verification, this subsystem exports model components to MLIR for hardware backend generation.

| Module | Purpose |
|--------|---------|
| `compile.py` | Full-weight MLIR export (FeedForward, Attention) |
| `compile_simple.py` | Constant-weight export for readability |
| `compile_complex.py` | Multi-block transformer with sequential/merged/scf_loop modes |
| `mlir_util.py` | Regex-based constant compression |

**Pipeline**: `torch.export` -> `torch_mlir.fx.export_and_import` -> Torch dialect MLIR

**Known limitation**: `torch_mlir` auto-unrolls Python loops, so `scf_loop` mode produces flat (merged-like) MLIR without `scf.for` constructs.

---

## 7. Data Flow

### 7.1 Inference with Verification

```
1. Load model weights to both CPU (pinned) and GPU
2. For each forward pass:
   a. GPU executes linear/matmul on compute_stream
   b. copy_stream waits on compute_stream, copies result to pinned CPU
   c. ThreadPool worker receives CPU copy, runs Freivalds check
   d. GPU proceeds to next operation (non-blocking)
   e. Non-matmul ops (activation, norm) run on CPU
3. At layer/step boundary: flush() waits for all pending verifications
4. If any verification fails:
   - fail_on_error=True: raise VerificationError
   - fail_on_error=False: log warning, continue
```

### 7.2 Memory Layout

| Memory Region | Contents |
|---------------|----------|
| GPU Global | Model weights, activations, intermediate results |
| CPU Pinned | Weight transposes, bias copies, GPU output copies |
| CPU Regular | Random vectors, Freivalds computation buffers |

---

## 8. Profiling System

`VerifyProfiler` collects per-operation records:

| Field | Type | Description |
|-------|------|-------------|
| `category` | str | "verify", "transfer", "verify_skip" |
| `op_name` | str | "linear_freivalds", "matmul_freivalds" |
| `duration_ms` | float | Wall-clock time |
| `shape` | tuple | Tensor dimensions |
| `tag` | str | Layer identifier (e.g., "block.3.attn.to_q") |
| `ok` | bool | Verification passed/failed |

**Export**: `profiler.export_csv("results.csv")`, `profiler.plot()` (matplotlib)

---

## 9. Testing Strategy

### 9.1 Test Levels

| Level | Files | What's Tested |
|-------|-------|---------------|
| Unit | `test_linear.py`, `test_zimage_verify_ops.py` | Freivalds algorithm, VerifyLinear forward |
| Module | `test_attn_layer.py`, `test_zimage_module_level.py` | Attention, FFN layer wrappers |
| Model | `test_llm_model.py` | Model creation, layer replacement |
| Integration | `test_compile.py` | MLIR compilation |
| E2E | `test_zimage_pipeline_fast.py`, `test_zimage_pipeline_slow.py` | Full pipeline inference |

### 9.2 Key Test Scenarios

- **Correct GPU output**: Verification should pass
- **Noise injection**: Inject controlled noise into GPU output, verify detection
- **Parametrized dimensions**: Multiple batch sizes, sequence lengths, hidden dims
- **Confidence levels**: Different `k` values for Freivalds iterations

---

## 10. Performance Characteristics

### 10.1 Overhead Budget

| Component | Overhead | Notes |
|-----------|----------|-------|
| GPU forward pass | ~0% | Unchanged from baseline |
| D2H copy (pinned) | 3-8% of layer time | PCIe bandwidth-bound |
| CPU Freivalds | ~5-15% of layer time | O(n^2 * k), runs in parallel |
| Synchronization | < 1% | Event-based, minimal |
| **Total** | **~5-15%** | Amortized across layers |

### 10.2 Scaling

- Overhead **decreases** with larger batch sizes (amortized copy cost)
- Overhead **increases** with more verification iterations (higher `k`)
- `verify_every_n > 1` trades confidence for throughput (adaptive mode)

---

## 11. File Structure

```
dt/
├── verified_llm/                  # LLM verification module
│   ├── verify_linear.py           # Freivalds algorithm + VerifyLinear
│   ├── attn_layer.py              # LlamaAttentionVerify
│   ├── mlp_layer.py               # LlamaMLPVerify
│   ├── sparse_attn_layer.py       # Sparse attention variant
│   ├── llm_model.py               # Model injection utilities
│   ├── eval.py                    # Evaluation entry point
│   ├── profiler.py                # Timing profiler
│   └── log_utils.py               # Logging
│
├── verified_diffusers/            # Diffusion model verification
│   ├── zimage/                    # Z-Image verified pipeline
│   │   ├── config.py              # VerifyConfig dataclass
│   │   ├── runtime.py             # VerifyRuntime (async executor)
│   │   ├── chunked_runtime.py     # ChunkedVerifyRuntime
│   │   ├── layers.py              # VerifyLinearModule, VerifyMatmul
│   │   ├── attention.py           # VerifiedZImageAttention
│   │   ├── mlp.py                 # VerifiedZImageFeedForward
│   │   ├── transformer_block.py   # VerifiedZImageTransformerBlock
│   │   ├── transformer.py         # VerifiedZImageTransformer2DModel
│   │   ├── pipeline.py            # VerifiedZImagePipeline
│   │   └── profiler.py            # VerifyProfiler
│   ├── compile.py                 # MLIR compilation (full weights)
│   ├── compile_simple.py          # MLIR compilation (const weights)
│   ├── compile_complex.py         # Multi-block MLIR compilation
│   ├── mlir_util.py               # MLIR utilities
│   └── hooks.py                   # PyTorch forward/backward hooks
│
├── tests/                         # Test suite
├── scripts/                       # Benchmarks and utilities
├── docs/                          # Documentation
├── dataset/                       # C4 test dataset
└── output/                        # Generated MLIR files
```

---

## 12. Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core tensor ops, CUDA streams, pinned memory |
| `transformers` | Llama model definitions |
| `diffusers` | Z-Image/Flux pipeline and modules |
| `torch-mlir` | MLIR compilation backend |
| `safetensors` | Model weight loading |
| `matplotlib` | Profiling visualization |

---

## 13. Future Directions

1. **Adaptive verification frequency**: Dynamically adjust `verify_every_n` based on historical failure rate
2. **Selective layer verification**: Only verify security-critical layers (e.g., final output projection)
3. **MLIR-based verification**: Lower Freivalds checks to MLIR for hardware-accelerated verification
4. **Multi-GPU support**: Extend runtime to handle distributed inference with per-device verification
5. **Fault logging and alerting**: Structured event log for verification failures with automated response
