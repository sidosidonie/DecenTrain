# Verified Neural Network Inference

Probabilistic verification of GPU-computed neural network inference using **SLALOM** (Tramer & Boneh, ICLR 2019) preprocessed Freivalds' algorithm. The GPU is treated as untrusted; every matmul/linear result is verified asynchronously on CPU.

Two model families are supported:
- **verified_llm** -- Llama, Qwen (2/2.5/3/3.5), Mistral-family LLMs
- **verified_diffusers** -- Z-Image diffusion transformer

## Quick Start

### 1. Requirements

- Python 3.12+
- CUDA 12.x + NVIDIA GPU (16 GB+ VRAM recommended)
- conda or pip

### 2. Install

```bash
# Clone
git clone <repo-url> && cd dt

# Create environment (conda recommended)
conda create -n verified-inference python=3.12 -y
conda activate verified-inference

# Install PyTorch with CUDA (adjust cu128 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify installation

```bash
# Unit tests (no model download, ~4 seconds)
pytest tests/test_zimage_module_level.py tests/test_zimage_verify_ops.py -v

# Threat model tests (no model download, ~4 seconds)
pytest tests/test_threat_model.py -v

# End-to-end LLM tests (downloads Qwen2.5-0.5B, ~1 GB, ~20 seconds)
RUN_E2E_LLM=1 pytest tests/test_e2e_llm.py -v
```

### 4. Run a model

```python
import torch
from verified_diffusers.zimage.config import VerifyConfig
from verified_llm.llm_model import create_llm_model

# Load model with verification enabled
model = create_llm_model(
    "Qwen/Qwen2.5-0.5B-Instruct",
    verify=True,
    config=VerifyConfig(enabled=True, profile_enabled=True),
    dtype=torch.float32,
    device="cuda",
    trust_remote_code=True,
)
model.eval()

# Run inference
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)

# Flush verification — raises RuntimeError if any matmul was corrupted
model._verify_runtime.flush()
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Running on Two Machines

The full LLM / diffusion stack runs in a single process. Cross-machine
deployment (1 trusted CPU coordinator + N untrusted GPU workers) is
described in [`docs/MULTI_MACHINE.md`](docs/MULTI_MACHINE.md); the
runnable slice today is [`examples/multi_machine_ffn.py`](examples/multi_machine_ffn.py)
— a coordinator + worker over TCP, outputs-only wire, real SLALOM
verification of a SwiGLU MLP.

The two roles have **different dependencies** — the GPU worker needs a
CUDA PyTorch build and the model libraries; the CPU coordinator needs
only a CPU-only PyTorch build plus the profiler/wire deps:

| | GPU worker (untrusted) | CPU coordinator / TEE (trusted) |
|---|---|---|
| PyTorch | CUDA build (`--index-url .../cu128`) | CPU-only build (`--index-url .../cpu`) |
| Requirements file | `requirements-gpu.txt` | `requirements-cpu.txt` |
| Extra libs | transformers, diffusers, safetensors, accelerate | pandas, matplotlib (transformers/safetensors only if it re-derives SLALOM `s_tilde` from local weights) |

**Machine A — GPU worker:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

python examples/multi_machine_ffn.py --role worker --bind 0.0.0.0:9100
```

**Machine B — CPU coordinator / TEE host:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt

python examples/multi_machine_ffn.py --role coordinator \
    --worker-host <machine-A-ip> --worker-port 9100 \
    --rounds 100 --hidden 4096 --inter 11008 --wire-dtype fp16
```

For a single-machine smoke test, run `python examples/multi_machine_ffn.py`
(default `--role loopback` — it forks its own worker). Useful flags:
`--inject-fault ...` (verify the coordinator catches a corrupted matmul),
`--pipeline` (overlap send/recv with compute), `--json-report out.json`.

(`requirements.txt` remains the single-machine, full-stack install.)

---

## How It Works

### Architecture

```
GPU (untrusted, full speed)          CPU (trusted, async)
──────────────────────────           ──────────────────────────
Forward pass runs completely:
  x@Wq, x@Wk, x@Wv                 ←─ D2H matmul outputs only
  q_norm, RoPE (GPU)
  Q@K^T                              ←─ D2H scores
  softmax (GPU)
  P@V                                ←─ D2H attn_out
  x@Wo                               ←─ D2H o_out
  silu(x@Wg) * x@Wu                  ←─ D2H gate, up
  x@Wd                               ←─ D2H down

GPU never waits for CPU.             CPU verification chain:
                                       SLALOM: y@s == x@s̃  (linear)
                                       Freivalds: ABr == Cr (matmul)
                                       Recompute: non-linears from
                                       verified chain data

                                     flush() → raise if any failure
```

### Verification Methods

| Operation | GPU | CPU Verification | Cost |
|-----------|-----|-----------------|------|
| Linear (Q/K/V/O, MLP) | `y = x @ W^T` | SLALOM preprocessed Freivalds | O(n*k) |
| Matmul (Q@K^T, P@V) | `C = A @ B` | Standard Freivalds | O(n^2*k) |
| Non-linear (softmax, silu, norm, RoPE) | Computed on GPU | CPU recomputes from chain | O(n) |

### SLALOM Preprocessing

Since model weights W are fixed at inference, `s_tilde = W^T @ s` is precomputed offline. Online verification becomes:

```
y @ s == x @ s_tilde     (two dot products, O(n*k) instead of O(n^2*k))
```

---

## Project Structure

```
.
├── verified_llm/                  # Verified LLM inference
│   ├── verify_linear.py           # SLALOM + Freivalds algorithms, VerifyLinear
│   ├── attn_layer.py              # LlamaAttentionVerify (CPU chain verification)
│   ├── mlp_layer.py               # LlamaMLPVerify (CPU chain verification)
│   └── llm_model.py               # create_llm_model(), duck-typed layer replacement
│
├── verified_diffusers/            # Verified diffusion inference
│   └── zimage/
│       ├── config.py              # VerifyConfig dataclass
│       ├── runtime.py             # VerifyRuntime (async verification engine)
│       ├── layers.py              # VerifyLinearModule, VerifyMatmul
│       ├── attention.py           # VerifiedZImageAttention (CPU chain)
│       ├── mlp.py                 # VerifiedZImageFeedForward (CPU chain)
│       ├── transformer_block.py   # VerifiedZImageTransformerBlock
│       ├── transformer.py         # VerifiedZImageTransformer2DModel
│       ├── pipeline.py            # VerifiedZImagePipeline
│       └── profiler.py            # VerifyProfiler (CSV/JSON/plot export)
│
├── tests/
│   ├── test_e2e_llm.py            # 4 end-to-end LLM tests
│   ├── test_threat_model.py       # 11 attack scenario tests
│   ├── test_zimage_module_level.py # 9 module-level tests
│   ├── test_zimage_verify_ops.py  # 2 attention/MLP verification tests
│   └── bench_qwen3_5_9b.py       # Qwen3.5-9B performance benchmark
│
├── docs/
│   ├── DESIGN.md                  # System design document
│   └── REPORT.md                  # Technical report (TEE survey, benchmarks)
│
└── output/                        # Profiling output (CSV, JSON, plots)
```

## Supported Models

| Model | Parameters | dtype | VRAM | Notes |
|-------|-----------|-------|------|-------|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | fp32 | ~2 GB | CI/test model |
| meta-llama/Llama-3.2-1B-Instruct | 1B | fp32 | ~4 GB | |
| Qwen/Qwen3-8B | 8B | bf16 | ~16 GB | |
| Qwen/Qwen3.5-9B | 9B | bf16 | ~17 GB | Hybrid (full + linear attention) |
| Tongyi-MAI/Z-Image | ~2B | fp16 | ~24 GB | Diffusion transformer |

Any HuggingFace causal LM with standard `q/k/v/o_proj` attention + `gate/up/down_proj` MLP is auto-detected via duck typing.

## Configuration

```python
from verified_diffusers.zimage.config import VerifyConfig

config = VerifyConfig(
    enabled=True,              # Toggle verification on/off
    freivalds_k=10,            # Random vectors (confidence)
    mse_threshold=1e-5,        # Max MSE for matmul verification
    verify_every_n=1,          # Verify every N-th op (1 = all)
    max_workers=2,             # CPU ThreadPool workers
    fail_on_error=True,        # Raise on verification failure
    profile_enabled=True,      # Enable timing profiler
)
```

**dtype-specific thresholds:**

| dtype | Recommended `mse_threshold` |
|-------|---------------------------|
| fp32 | 1e-5 |
| fp16 / bf16 | 5e-3 |
| fp8 | 1e-2 |
| int8 (quantized) | 5e-2 |

## Running Tests

```bash
# All tests (requires CUDA GPU)
RUN_E2E_LLM=1 pytest tests/ -v

# Fast tests only (no model download)
pytest tests/test_zimage_module_level.py tests/test_zimage_verify_ops.py tests/test_threat_model.py -v

# Qwen3.5-9B benchmark (requires ~17 GB VRAM, downloads ~18 GB model)
python tests/bench_qwen3_5_9b.py
```

## Benchmarks

### Qwen3.5-9B (RTX 5090, bf16)

| Metric | Origin | Verified | Overhead |
|--------|--------|----------|----------|
| Forward pass | 137 ms | 1,950 ms | 14.3x |
| Generation (50 tok) | 2,257 ms | 114,723 ms | 50.8x |
| Token equivalence | — | — | PASS |

### Z-Image (A100, fp32)

| Metric | Origin | Verified | Overhead |
|--------|--------|----------|----------|
| 1-step inference | 7,059 ms | 16,115 ms | 2.3x |

## References

- Tramer, F. & Boneh, D. "SLALOM: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware." ICLR, 2019. [arXiv:1806.03287](https://arxiv.org/abs/1806.03287)
- Freivalds, R. "Probabilistic Machines Can Use Less Running Time." IFIP Congress, 1977.
- See `docs/REPORT.md` for the full technical report (TEE survey, quantization evaluation).
- See `docs/DESIGN.md` for the system design document.
