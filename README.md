# Verified Neural Network Inference

Probabilistic verification of GPU-computed neural network inference using **Freivalds' algorithm**. The GPU is treated as untrusted; every matmul/linear result is verified asynchronously on CPU (future: TEE).

Two model families are supported:
- **verified_llm** -- Llama-family LLMs (attention + MLP)
- **verified_diffusers** -- Z-Image diffusion transformer + Flux

## Quick Start

### 1. Environment

Tested: Python 3.12, CUDA 12.x, conda 25.5.1.

```bash
conda env create -f env.yml
conda activate verified-llm
pip install -r requirements.txt
```

### 2. Patch transformers (one-time)

Export `LlamaAttention`, `LlamaMLP`, `apply_rotary_pos_emb` from transformers:

```bash
# Find your transformers install path
python -c "import transformers; print(transformers.__file__)"
```

Edit `transformers/models/llama/modeling_llama.py`, add to `__all__`:

```python
__all__ = [
    ...,
    "LlamaAttention",
    "LlamaMLP",
    "apply_rotary_pos_emb",
]
```

### 3. Models

- LLM: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- Diffusion: [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) (Z-Image pipeline)

### 4. Dataset (LLM only)

```bash
git lfs pull   # downloads c4 test data in dataset/c4
```

---

## Running

### Unit Tests

```bash
# All tests
pytest tests/ -v

# Diffusers module-level tests (no model download needed)
pytest tests/test_zimage_module_level.py tests/test_zimage_verify_ops.py -v

# LLM linear / MLP / Freivalds tests
pytest tests/test_linear.py -v

# LLM model-level test (requires Llama download)
pytest tests/test_llm_model.py -v
```

### LLM End-to-End Evaluation

```bash
mkdir -p logs

# Perplexity evaluation with verification
python verified_llm/eval.py

# Noise injection sweep
chmod +x ppl.sh
./ppl.sh
# Output: logs/ppl-llama-noise-${noise_scale}-limit-${limit_samples}-loss
```

### Diffusion (Z-Image) End-to-End

```bash
# Compare verified vs unverified inference (timing + profiling)
python -m verified_diffusers.zimage.run_slow_e2e_compare

# With options
python -m verified_diffusers.zimage.run_slow_e2e_compare \
    --num_inference_steps 4 \
    --num_runs 3 \
    --output_dir ./perf_out \
    --save_images True
```

Profiling output (CSV + plots) is saved to `output/zimage_verify_profile/` by default.

### Diffusion (Flux)

```bash
python verified_diffusers/verified_flux.py
```

---

## Project Structure

```
.
├── verified_llm/                  # Verified LLM inference
│   ├── verify_linear.py           # VerifyLinear + Freivalds algorithms
│   ├── attn_layer.py              # LlamaAttentionVerify (VerifyRuntime-backed)
│   ├── mlp_layer.py               # LlamaMLPVerify (VerifyRuntime-backed)
│   ├── llm_model.py               # Model creation: create_llm_model()
│   ├── eval.py                    # Perplexity evaluation entry point
│   ├── sparse_attn_layer.py       # Sparse attention variant
│   └── legacy/                    # Old synchronous code (deprecated)
│
├── verified_diffusers/            # Verified diffusion inference
│   ├── zimage/                    # Z-Image transformer pipeline
│   │   ├── config.py              # VerifyConfig dataclass
│   │   ├── runtime.py             # VerifyRuntime (async verification engine)
│   │   ├── layers.py              # VerifyLinearModule, VerifyMatmul
│   │   ├── attention.py           # VerifiedZImageAttention (manual verified attention)
│   │   ├── transformer_block.py   # VerifiedZImageTransformerBlock (incl. adaLN)
│   │   ├── transformer.py         # VerifiedZImageTransformer2DModel
│   │   ├── mlp.py                 # VerifiedZImageFeedForward
│   │   ├── pipeline.py            # VerifiedZImagePipeline wrapper
│   │   ├── profiler.py            # VerifyProfiler (timing + CSV export)
│   │   └── run_slow_e2e_compare.py
│   ├── verified_flux.py           # Flux model verification
│   └── hooks.py                   # Diffusers pipeline hooks
│
├── tests/                         # pytest test suite
│   ├── conftest.py                # Environment shims (torchvision, flash_attn)
│   ├── test_linear.py             # VerifyLinear + Freivalds tests
│   ├── test_llm_model.py          # LLM model creation test
│   ├── test_zimage_module_level.py # VerifyLinearModule, VerifyMatmul, corruption detection
│   ├── test_zimage_verify_ops.py  # Attention + MLP verified vs origin comparison
│   └── test_zimage_pipeline_*.py  # Pipeline integration tests
│
├── dataset/c4/                    # C4 validation dataset (git lfs)
├── docs/DESIGN.md                 # Detailed design document
├── env.yml                        # Conda environment
└── requirements.txt               # pip dependencies
```

## Architecture

```
GPU (untrusted)              CPU (trusted / TEE)
─────────────────            ──────────────────────────
 compute_stream               ThreadPoolExecutor
   │                            │
   ├─ F.linear(x, W)           │
   │     │                      │
   │     ├─── copy_stream ──────┤ copy x, y to pinned CPU
   │     │                      │
   ├─ matmul(Q, K^T)           ├─ Freivalds: ABr vs Cr
   │     │                      │   MSE < threshold? ──→ OK / FAIL
   │     ├─── copy_stream ──────┤
   │     │                      │
   ├─ softmax (GPU)             ├─ Recompute softmax on CPU
   │     │                      │   MSE < threshold? ──→ OK / FAIL
   │     ├─── copy_stream ──────┤
   │     │                      │
   ├─ matmul(attn, V)          ├─ Freivalds verify
   │     ...                    │   ...
   │                            │
   └── (continues)              └── flush() → raise if any failure
```

Key properties:
- **Zero `torch.cuda.synchronize()`** in the verified forward path
- GPU compute and CPU verification run in parallel via `VerifyRuntime`
- Copy is overlapped using a dedicated `copy_stream` + pinned memory
- Verification results are collected asynchronously; `runtime.flush()` at end

## Configuration

```python
from verified_diffusers.zimage.config import VerifyConfig

config = VerifyConfig(
    enabled=True,                    # Toggle verification on/off
    freivalds_k=8,                   # Random vectors (confidence: 1 - 2^{-k})
    mse_threshold=1e-5,              # Max MSE for linear/matmul
    elementwise_mse_threshold=1e-4,  # Max MSE for softmax/silu
    verify_every_n=1,                # Verify every N-th op (1 = all)
    max_workers=2,                   # CPU ThreadPool workers
    max_verify_tensor_numel=2_000_000,  # Skip very large tensors
    fail_on_error=True,              # Raise on verification failure
    profile_enabled=True,            # Enable timing profiler
)
```

## Verified Operations

| Operation | Method | Location |
|-----------|--------|----------|
| nn.Linear | Freivalds (submit_linear) | All projection layers |
| Q @ K^T | Freivalds (submit_matmul) | Attention QK |
| attn_probs @ V | Freivalds (submit_matmul) | Attention KV |
| softmax | CPU recompute + MSE (submit_elementwise) | After QK scaling |
| silu | CPU recompute + MSE (submit_elementwise) | MLP activation |
| adaLN_modulation | Freivalds (via VerifyLinearModule) | Transformer blocks |

## Benchmarks

### Z-Image (A100 + 20-core CPU)

| Prompt | Origin (1 step) | Verified (1 step) | Overhead |
|--------|------------------|--------------------|----------|
| Short  | 6,959 ms         | 15,948 ms          | 2.3x     |
| Long   | 7,059 ms         | 16,115 ms          | 2.3x     |

### Flux.1-schnell (A100, 30 steps)

| Mode | Total Time |
|------|------------|
| Origin | 117,804 ms |
| Verified | 403,913 ms |

---

## References

- [Freivalds' Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Freivalds%27_algorithm)
- See `docs/DESIGN.md` for the full design document
