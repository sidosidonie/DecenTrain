# Multi-Machine Attention & Zimage Examples — Design

> Three new standalone, runnable multi-machine examples that mirror the
> existing `examples/multi_machine_ffn.py` pattern: coordinator + worker over
> TCP, real SLALOM verification, outputs-only wire. Targets a single
> Llama-style attention block, a single Z-Image diffusion attention block,
> and a self-contained "mini-zimage" stack of N transformer blocks.
>
> Companion to: `docs/superpowers/specs/2026-05-11-multi-machine-ffn-example-design.md`.

Status: design, ready for implementation plan.

---

## 1. Goal & Non-Goals

### Goal

Add three new files under `examples/`, plus one shared helper module:

1. `examples/_multi_machine_common.py` — wire framing, SLALOM helpers,
   per-tensor packers, RoPE, RMSNorm, summary scaffolding, loopback
   launcher. Used by the three new examples; the existing FFN example
   stays self-contained.
2. `examples/multi_machine_attn_llama.py` — single Llama/Qwen-style
   attention block (q/k/v + RoPE + softmax + o), GQA-configurable.
3. `examples/multi_machine_attn_zimage.py` — single Z-Image diffusion
   attention block (q/k/v + per-head RMSNorm + complex-cis RoPE +
   softmax + o, no causal mask, no GQA).
4. `examples/multi_machine_zimage.py` — `N`-block mini-zimage built
   from the same primitives (RMSNorm + zimage-attn + SwiGLU MLP per
   block, with residuals). Self-contained, **does not** wrap the
   diffusers `ZImageTransformer2DModel`.

Each new file is one CLI binary with `--role {loopback,worker,coordinator}`,
matching FFN's launch ergonomics.

### Non-Goals

- **Not** wrapping the real diffusers `ZImageTransformer2DModel`. The full
  model has refiners, x_embedder, t_embedder, cap_embedder,
  adaLN_modulation, noise/clean splits — patching all of them is its own
  project. The "full diffusers wrap" gets a separate spec if/when needed.
- **Not** a multi-worker / weight-sharded example.
- **Not** KV-cache / autoregressive decoding for the Llama attn example
  (single forward only).
- **Not** a refactor of `examples/multi_machine_ffn.py` to use the new
  shared module. FFN stays untouched so existing perf numbers and
  `examples/run_*.sh` keep working.
- **Not** a Freivalds-based verification of the inner attention matmuls
  (`q@k^T`, `probs@v`). The user explicitly chose the FFN-style
  approach: SLALOM the linears, recompute the non-linear core on the
  coordinator. Freivalds-vs-recompute is a future optimization.

---

## 2. Architecture

Same coordinator/worker shape as FFN. Differences are concentrated in
(a) what gets shipped on the wire and (b) how the coordinator reconstructs
the trusted intermediate state.

```
┌─ Coordinator (trusted CPU) ──────────────────────────────┐
│  • CPU-side weights for every linear in the example      │
│  • Per-linear SLALOM state: s_i, s_tilde_i = W_iᵀ @ s_i  │
│  • Per-block CPU chain: rmsnorm, q/k norm, RoPE, softmax │
│  • Verifier thread pool                                  │
│  • Metrics accumulator                                   │
└────────────────────────────┬─────────────────────────────┘
                             │ TCP
                             ▼
┌─ Worker (untrusted GPU) ─────────────────────────────────┐
│  • GPU weights for every linear (same seed → identical)  │
│  • Full forward; D2H of every matmul output back to wire │
│  • Optional pipelined / streaming sender                 │
│  • Optional fault injector                               │
└──────────────────────────────────────────────────────────┘
```

**Verification chain** (mirrors FFN's "verify linears, redo non-linears"):

For each attention block:
1. Worker sends `q_raw`, `k_raw`, `v_raw`, `o_raw` (= block output before
   any post-residual ops). Inner `attn_out` is **not** sent.
2. Coordinator SLALOM-verifies `q_raw`, `k_raw`, `v_raw` against the
   block's CPU input.
3. Coordinator recomputes the non-linear core on CPU: optional
   reshape/per-head norm, RoPE, `q@k^T * scale + mask`, softmax (fp32),
   `probs @ v`, reshape → `attn_out_cpu`.
4. Coordinator SLALOM-verifies `o_raw` against `attn_out_cpu`.

For each FFN block (zimage example): identical to the standalone FFN
example — verify `w1_out`, `w3_out` against the block's CPU input,
recompute `silu(w1_out) * w3_out` on CPU, verify `w2_out` against that.

For the multi-block zimage example, blocks chain: block `b`'s verified
output (after the residual additions, computed on the coordinator from
verified intermediates) becomes block `b+1`'s CPU chain input.

---

## 3. Shared Module: `examples/_multi_machine_common.py`

One module, ~300 lines. Existing FFN file does **not** import from it.

### 3.1 Wire framing

Identical to FFN.

```python
def recv_exactly(sock, n) -> bytes
def send_msg(sock, msg_type, body) -> int
def recv_msg(sock) -> tuple[int, bytes]

class WireProtocolError(RuntimeError): pass

MSG_LOAD_REQ      = 1
MSG_LOAD_ACK      = 2
MSG_FORWARD_REQ   = 3
MSG_TENSOR        = 4   # generalized MSG_ACTIVATION
MSG_FORWARD_DONE  = 5
MSG_CLOSE         = 6
```

### 3.2 Dtype tables

Identical to FFN. `DTYPE_FP16=2`, `DTYPE_FP32=1`, `_TORCH_DTYPE`,
`_NUMPY_DTYPE`, `_DTYPE_SIZE`, `_DTYPE_NAME`, `_NAME_TO_DTYPE`.

### 3.3 Generic tensor packer

```python
def pack_tensor(request_id: int, op_tag: int, tensor: torch.Tensor,
                wire_dtype_id: int) -> bytes:
    """Header: <Q H B B  shape[ndim] (I*ndim)>  + payload.
       request_id (u64) + op_tag (u16) + dtype_id (u8) + ndim (u8)."""

def unpack_tensor(body: bytes) -> dict
```

Two changes vs FFN's `pack_activation`:
- `op_tag` is `H` (uint16) instead of `B` (uint8). Required by zimage
  (4096 blocks × 7 op kinds packed into one tag); the attn examples use
  the low byte only.
- Renamed `MSG_ACTIVATION` → `MSG_TENSOR` since some tensors (e.g. block
  outputs in zimage) aren't strictly intermediate activations.

### 3.4 SLALOM helpers

Same as FFN, lifted verbatim into the shared module:

```python
SLALOM_K = 10
S_GENERATOR_SEED = 0xDEADBEEF

def make_s(out_dim, k, seed) -> torch.Tensor       # (out_dim, k) fp32 cpu
def precompute_s_tilde(weight, s) -> torch.Tensor  # (in_dim, k)  fp32 cpu
def slalom_verify(x, y, s, s_tilde) -> float       # mse
def slalom_verify_safe(x, y, s, s_tilde) -> float  # inf on NaN/Inf
```

### 3.5 RoPE helpers

Two RoPE flavors live here, both deterministic and parameter-free:

```python
# Llama / Qwen: cos/sin pair, applies to (B, H, S, D) tensors
def precompute_rope_cos_sin(head_dim: int, max_seq: int,
                            base: float = 500000.0,
                            device="cpu", dtype=torch.float32
                            ) -> tuple[torch.Tensor, torch.Tensor]
def apply_rope_llama(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]

# Z-Image: complex freqs_cis, applies to (B, S, H, D) tensors
def precompute_zimage_freqs_cis(head_dim: int, max_seq: int,
                                theta: float = 10000.0
                                ) -> torch.Tensor   # complex (S, D/2)
def apply_rotary_emb_zimage(x: torch.Tensor,
                            freqs_cis: torch.Tensor) -> torch.Tensor
```

The Llama variant matches HuggingFace's `apply_rotary_pos_emb` (rotate
pairs of channels via real cos/sin). The zimage variant matches
`verified_diffusers/zimage/attention.py:30-43` (rotate pairs as complex
multiplication).

Both implementations have a CPU and a CUDA path (the CUDA path is just
"call the CPU formula on whatever device the inputs live on" — they
contain no device-specific kernels).

### 3.6 RMSNorm CPU helper

```python
def rmsnorm_cpu(x: torch.Tensor, weight: Optional[torch.Tensor],
                eps: float, *, scale_offset: float = 0.0) -> torch.Tensor
```

`scale_offset=0.0` for zimage / generic; `scale_offset=1.0` for the
Qwen3-style `(1.0 + weight)` scaling. Returns fp32.

### 3.7 RoundMetricsBase

```python
@dataclass
class RoundMetricsBase:
    request_id: int
    coord_send_t: float = 0.0
    gpu_forward_t: float = 0.0
    wire_recv_t: float = 0.0
    cpu_verify_t: float = 0.0
    end_to_end_t: float = 0.0
    bytes_sent: int = 0
    bytes_recv: int = 0
    bytes_recv_predicted: int = 0
    recv_tensors: dict = field(default_factory=dict)  # op_tag -> {shape,dtype,bytes}
    mse: dict[int, float] = field(default_factory=dict)  # op_tag -> mse
    cpu_verify_per_op_t: dict[int, float] = field(default_factory=dict)
    ok: bool = True
```

Each example subclasses or extends this with example-specific fields
(e.g. `attn_recompute_t` for the attention examples).

### 3.8 Summary formatter factory

```python
def format_summary(rounds, cfg, *, warmup, k, pipelined, link_gbps,
                   op_name_map: dict[int, str],
                   extra_config_lines: list[str] = ()) -> str
```

Reuses FFN's structure (end-to-end ms, phase timings, phase breakdown,
wire-rate breakdown, wire bytes predicted-vs-measured, per-tensor wire
record, wire estimate / link efficiency, per-op MSE p95). Each example
passes its own `op_name_map` and any extra header lines.

### 3.9 Loopback launcher factory

```python
def launch_loopback(this_file: str, args, extra_worker_argv: list[str]
                    ) -> int
```

Spawns `python this_file --role worker --bind 127.0.0.1:<free-port> ...`
with the example's extra args, waits for the port, then runs the
coordinator. Same subprocess management / SIGTERM behavior as FFN.

### 3.10 Default-threshold helper

```python
def default_slalom_threshold(wire_dtype_id: int, in_dim: int, *,
                              floor: float = 1e-3,
                              fp16_slope: float = 2e-6) -> float
```

`fp32 → 1e-3`, `fp16 → max(floor, in_dim * fp16_slope)`. The o-projection
in attention examples uses a looser slope (`4e-6`) because its CPU input
is itself derived from worker-quantized q/k/v through fp32 softmax.

---

## 4. `examples/multi_machine_attn_llama.py`

### 4.1 Config (CLI args + dataclass)

```python
@dataclass
class AttnLlamaConfig:
    hidden: int = 4096
    heads: int = 32
    kv_heads: int = 32       # set lower for GQA, e.g. 8
    head_dim: int = 128      # = hidden // heads, asserted
    batch: int = 1
    seq: int = 512
    rope_base: float = 500000.0
    wire_dtype: int = DTYPE_FP16
    weight_seed: int = 0xC0FFEE
```

Asserts: `hidden % heads == 0`, `hidden // heads == head_dim`,
`heads % kv_heads == 0`. `num_kv_groups = heads // kv_heads`.

### 4.2 Weights

Four `nn.Linear`, no bias. Built with the same recipe as FFN
(`make_weights`): CPU `torch.Generator` seeded with
`(weight_seed + offset)`, std=0.02, then `.to(device, dtype)`. Offsets:
`q=0, k=1, v=2, o=3`.

```
q_proj : hidden     → heads * head_dim
k_proj : hidden     → kv_heads * head_dim
v_proj : hidden     → kv_heads * head_dim
o_proj : heads*hd   → hidden
```

### 4.3 Worker compute (GPU, single forward)

```python
q = q_proj(x); k = k_proj(x); v = v_proj(x)
q = q.view(B, S, heads, head_dim).transpose(1, 2)             # (B,H,S,D)
k = k.view(B, S, kv_heads, head_dim).transpose(1, 2)
v = v.view(B, S, kv_heads, head_dim).transpose(1, 2)
q, k = apply_rope_llama(q, k, cos, sin)
k = repeat_kv(k, num_kv_groups); v = repeat_kv(v, num_kv_groups)
scores = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
scores = scores + causal_mask
probs = softmax(scores, dim=-1, dtype=fp32).to(dtype)
attn_out = (probs @ v).transpose(1, 2).reshape(B, S, hidden)
output = o_proj(attn_out)
```

Sends back: `OP_Q=1, OP_K=2, OP_V=3, OP_O=4` (= `output`); plus
`MSG_FORWARD_DONE`. Does **not** send `attn_out`.

### 4.4 Coordinator verify (CPU fp32)

```python
# 1. SLALOM on linears
mse_q = slalom_verify_safe(x_cpu, q_cpu, s_q, s_tilde_q)
mse_k = slalom_verify_safe(x_cpu, k_cpu, s_k, s_tilde_k)
mse_v = slalom_verify_safe(x_cpu, v_cpu, s_v, s_tilde_v)

# 2. Recompute attention on CPU using verified q/k/v
q_h = q_cpu.view(B, S, heads, head_dim).transpose(1, 2)
k_h = k_cpu.view(B, S, kv_heads, head_dim).transpose(1, 2)
v_h = v_cpu.view(B, S, kv_heads, head_dim).transpose(1, 2)
q_h, k_h = apply_rope_llama(q_h, k_h, cos_cpu, sin_cpu)
k_h = repeat_kv(k_h, num_kv_groups); v_h = repeat_kv(v_h, num_kv_groups)
scores = q_h @ k_h.transpose(-2, -1) * scale + causal_mask_cpu
probs = softmax(scores, dim=-1, dtype=fp32)
attn_out_cpu = (probs @ v_h).transpose(1, 2).reshape(B, S, hidden)

# 3. SLALOM on output linear
mse_o = slalom_verify_safe(attn_out_cpu, output_cpu, s_o, s_tilde_o)

ok = (mse_q ≤ thr_qkv) and (mse_k ≤ thr_qkv) and (mse_v ≤ thr_qkv) \
     and (mse_o ≤ thr_o)
```

### 4.5 Pipelining (`--pipeline`)

Worker side: same background-sender thread pattern as FFN, queues each
of `q,k,v,o,DONE` as soon as packed.

Coordinator side: as q, k, v arrive, submit their SLALOM checks to the
pool. When v has arrived, kick off a single chained future
`recompute-attn → SLALOM-on-o`. When o arrives, that future's last leg
runs.

`cpu_verify_t` measures `last_future.completed - first_future.submitted`,
overlapping `wire_recv_t` (so `sum_pct > 100%`, same as FFN).

### 4.6 Fault injection (`--inject-fault`)

| Value | What worker does | Caught by |
|-------|------------------|-----------|
| `none` | nothing | — |
| `flip_v` | returns `-v` | `mse_v` SLALOM |
| `scale_o` | returns `output * 1.01` | `mse_o` SLALOM |
| `drop_softmax` | recomputes `output` from `o_proj((scores @ v).flatten...)`, skipping softmax | `mse_o` SLALOM (CPU recompute uses softmax) |
| `drop_rope` | skips `apply_rope` on the worker | `mse_o` SLALOM (CPU recompute uses RoPE) |

Tests (Section 7) assert each fault produces `mse > 10 × threshold`.

### 4.7 Threshold

`thr_qkv = default_slalom_threshold(wire_dtype, hidden)`.
`thr_o = default_slalom_threshold(wire_dtype, hidden, fp16_slope=4e-6)`.

The o-input is the recomputed `attn_out_cpu`, which is fp32 but built
from worker-fp16 q/k/v. Its noise floor is roughly the q/k/v wire
quantization, amplified by the attention math. We start with `4e-6`
slope and validate empirically with `--inject-fault none` + `--rounds
100` to confirm comfortable headroom over the noise floor.

---

## 5. `examples/multi_machine_attn_zimage.py`

### 5.1 Config

```python
@dataclass
class AttnZimageConfig:
    dim: int = 1536
    heads: int = 12              # head_dim = dim // heads
    head_dim: int = 128
    batch: int = 2
    seq: int = 1024
    qk_norm: str = "rms"         # "rms" or "none"
    rope_theta: float = 10000.0
    wire_dtype: int = DTYPE_FP16
    weight_seed: int = 0xC0FFEE
```

No causal mask (diffusion attention is bidirectional). No GQA
(`kv_heads == heads`). `head_dim` must be even (RoPE pairs channels).

### 5.2 Weights

Four `nn.Linear` (no bias) plus two RMSNorm scale weights (per-head_dim,
fp32 on both sides, generated from seed offsets `q_norm=4, k_norm=5`).

```
q_proj : dim → heads * head_dim
k_proj : dim → heads * head_dim
v_proj : dim → heads * head_dim
o_proj : heads * head_dim → dim
norm_q : (head_dim,) RMSNorm scale
norm_k : (head_dim,) RMSNorm scale
```

### 5.3 Worker compute

```python
q = q_proj(x); k = k_proj(x); v = v_proj(x)
q = q.unflatten(-1, (heads, head_dim))      # (B, S, H, D)
k = k.unflatten(-1, (heads, head_dim))
v = v.unflatten(-1, (heads, head_dim))
if qk_norm == "rms":
    q = norm_q(q); k = norm_k(k)
q = apply_rotary_emb_zimage(q, freqs_cis)   # uses (B, S, H, D)
k = apply_rotary_emb_zimage(k, freqs_cis)
q_t = q.permute(0, 2, 1, 3)                 # (B, H, S, D)
k_t = k.permute(0, 2, 1, 3)
v_t = v.permute(0, 2, 1, 3)
scores = q_t @ k_t.transpose(2, 3) * (head_dim ** -0.5)
probs = softmax(scores, dim=-1, dtype=fp32).to(dtype)
attn_out = (probs @ v_t).permute(0, 2, 1, 3).flatten(2, 3)
output = o_proj(attn_out)
```

Sends back: `OP_Q, OP_K, OP_V, OP_O` (= `output`).

### 5.4 Coordinator verify

Same shape as Llama variant, with the differences:
- Apply zimage RMSNorm (per head_dim) on q and k after reshape.
- Apply complex-cis RoPE.
- No causal mask; no `repeat_kv`.

### 5.5 Pipelining: same shape as Llama variant.

### 5.6 Fault injection

`none`, `flip_v`, `scale_o`, `drop_softmax`, `drop_rope`,
`drop_qk_norm` (skip the RMSNorm on q/k — caught by `mse_o` since the
coordinator's recompute applies the norm).

### 5.7 Threshold: same formula as Llama variant.

### 5.8 Smoke-test default

`dim=64 heads=4 batch=4 seq=64 rounds=20 warmup=2 device=cpu` — matches
`tests/test_zimage_verify_ops.py` shape, completes in <5s.

---

## 6. `examples/multi_machine_zimage.py`

### 6.1 Scope

Self-contained "mini-zimage": a stack of `N` transformer blocks built
from the same primitives as the attn-zimage example for the attention
sub-block and the FFN example for the FFN sub-block. Two
RMSNorm-residual fences per block.

**Does not** wrap the diffusers `ZImageTransformer2DModel`. The full
diffusers model has refiners, embedders, adaLN modulation, and
noise/clean splits — out of scope for this design.

### 6.2 Config

```python
@dataclass
class ZimageConfig:
    dim: int = 1536
    heads: int = 12
    head_dim: int = 128
    ffn_inter: int = 4096
    n_layers: int = 12
    batch: int = 2
    seq: int = 256
    qk_norm: str = "rms"
    rope_theta: float = 10000.0
    wire_dtype: int = DTYPE_FP16
    weight_seed: int = 0xC0FFEE
```

Smoke-test default: `dim=64 heads=4 ffn_inter=256 n_layers=2 batch=2
seq=64`.

### 6.3 Weights

Per block `b ∈ [0, N)`, per-block stride 16 (headroom for future ops):

```
attention_norm1[b] : RMSNorm  (dim,)        seed = base + 16b + 0
attn[b].q_proj    : dim → heads*head_dim   seed = base + 16b + 1
attn[b].k_proj    : dim → heads*head_dim   seed = base + 16b + 2
attn[b].v_proj    : dim → heads*head_dim   seed = base + 16b + 3
attn[b].norm_q    : (head_dim,)             seed = base + 16b + 4
attn[b].norm_k    : (head_dim,)             seed = base + 16b + 5
attn[b].o_proj    : heads*head_dim → dim   seed = base + 16b + 6
ffn_norm1[b]      : RMSNorm  (dim,)         seed = base + 16b + 7
ffn[b].w1         : dim → ffn_inter        seed = base + 16b + 8
ffn[b].w3         : dim → ffn_inter        seed = base + 16b + 9
ffn[b].w2         : ffn_inter → dim        seed = base + 16b + 10
```

### 6.4 Op tag namespace

```python
op_tag = (block_idx << 4) | op_kind   # uint16, from pack_tensor
op_kind:
  Q  = 1
  K  = 2
  V  = 3
  O  = 4
  W1 = 5
  W3 = 6
  W2 = 7
```

Worker sends `7N + 1` messages per round (7 tensors per block + 1
DONE). Total tag space: 4096 blocks × 7 op kinds, plenty of headroom.

### 6.5 Worker compute

```python
x_in = input
for b in range(N):
    # ── Attention sub-block ──
    x_norm = attention_norm1[b](x_in)
    q,k,v = projections(x_norm)
    q,k = qk_norm + RoPE
    attn_out = softmax(q @ kᵀ * scale) @ v
    o = o_proj(attn_out.reshape(B, S, dim))
    send(Q,K,V,O for block b)         # streamed if --stream
    x_after_attn = x_in + o

    # ── FFN sub-block ──
    h = ffn_norm1[b](x_after_attn)
    w1_out = w1(h); w3_out = w3(h)
    gated = silu(w1_out) * w3_out
    w2_out = w2(gated)
    send(W1,W3,W2 for block b)        # streamed if --stream
    x_in = x_after_attn + w2_out

send(DONE)
```

Two send modes:
- **Sequential** (default off): ship everything after the last block
  finishes. Simple, high latency.
- **Streaming** (`--stream`, default on): background sender thread; each
  block's tensors are queued the moment they're computed, so wire
  transfer overlaps the next block's GPU compute.

### 6.6 Coordinator verify

Receive loop tags incoming tensors by `(block_idx, op_kind)`. As soon as
block `b`'s seven tensors are all received and `x_in_for_b` is ready
(`b == 0` initially, otherwise `x_in_for_b` was just produced by block
`b-1`'s verify task), kick off block `b`'s verify task in the pool:

```python
def verify_block_b(b, x_in_b, q,k,v,o, w1,w3,w2):
    # Attention sub-block
    x_norm = rmsnorm_cpu(x_in_b, attention_norm1[b].weight, eps)
    mse_q = slalom_verify_safe(x_norm, q, s_q[b], s_tilde_q[b])
    mse_k = slalom_verify_safe(x_norm, k, s_k[b], s_tilde_k[b])
    mse_v = slalom_verify_safe(x_norm, v, s_v[b], s_tilde_v[b])
    attn_out_cpu = recompute_attn_zimage(q, k, v, freqs_cis, ...)
    mse_o = slalom_verify_safe(attn_out_cpu, o, s_o[b], s_tilde_o[b])
    x_after_attn = x_in_b + o

    # FFN sub-block
    h = rmsnorm_cpu(x_after_attn, ffn_norm1[b].weight, eps)
    mse_w1 = slalom_verify_safe(h, w1, s_w1[b], s_tilde_w1[b])
    mse_w3 = slalom_verify_safe(h, w3, s_w3[b], s_tilde_w3[b])
    gated_cpu = silu(w1) * w3
    mse_w2 = slalom_verify_safe(gated_cpu, w2, s_w2[b], s_tilde_w2[b])
    x_in_for_next = x_after_attn + w2

    record mse[(b, op_kind)] for each op_kind
    return x_in_for_next
```

Block `b+1`'s task is submitted with a dependency on block `b`'s task's
result. So block-to-block computation is serialized on CPU but each
block's verify overlaps with the worker's compute and wire transfer of
later blocks.

### 6.7 Per-round metrics

Subclass `RoundMetricsBase`:

```python
@dataclass
class ZimageRoundMetrics(RoundMetricsBase):
    cpu_verify_per_block_t: list[float] = field(default_factory=list)
    bytes_recv_per_block:   list[int]   = field(default_factory=list)
    # mse: dict[(int, int) -> float] inherited  (block_idx, op_kind)
```

Summary reports per-block-mean times (line per block at `--verbose`,
otherwise just min/median/max across blocks) and the max-mse-per-op-kind
across all blocks.

### 6.8 Fault injection

Same set as the attn examples plus `--fault-block N` to target a
specific block. Default fault-block is `0`.

| Fault | Caught by |
|-------|-----------|
| `flip_v` (block b) | `mse_v[b]` |
| `scale_o` (block b) | `mse_o[b]` |
| `drop_softmax` (block b) | `mse_o[b]` |
| `drop_rope` (block b) | `mse_o[b]` |
| `drop_qk_norm` (block b) | `mse_o[b]` |
| `flip_w1` (block b) | `mse_w1[b]` |
| `scale_w2` (block b) | `mse_w2[b]` |
| `drop_silu` (block b) | `mse_w2[b]` |

A single-block fault must propagate into block `b+1`'s `mse` (because
the coordinator uses verified data, but the worker uses faulty data, so
their chain inputs diverge from block `b+1` onward) — the test in
Section 7 asserts at least block `b` and `b+1` both fail.

### 6.9 Threshold

`thr_per_op = max(1e-3, dim * 6e-6)` for fp16. Slope is slightly looser
than the per-block attn examples (`4e-6`) because errors compound across
the chain before each per-linear SLALOM check.

---

## 7. Testing

New file: `tests/test_multi_machine_examples.py`. Parametrized over:

- example ∈ `{ffn (sanity), attn_llama, attn_zimage, zimage}`
- fault ∈ `{none, flip_v, scale_o}` (plus `flip_y1` / `drop_silu` for
  ffn / zimage where applicable)

For each combination, run the example as a subprocess in `--role
loopback --device cpu` with the smoke-test config (small dims, few
rounds, fp16 wire), capture the JSON report (`--json-report`), and
assert:

1. **`fault == none`**: `rounds passed == rounds total`. All per-op MSE
   < threshold.
2. **`fault != none`**: at least one round's `ok == False`. The
   triggered op's MSE exceeds threshold by ≥10× (so we're not skating
   the boundary).
3. **Bytes prediction**: `bytes_recv == bytes_recv_predicted`
   round-trip exactly (sanity for the wire format).
4. **Determinism**: re-run same `--weight-seed` and same
   `--input-seed-start` → byte-identical mse values across two
   consecutive runs.
5. **Zimage block propagation**: `--inject-fault scale_o --fault-block
   3` produces `mse_o[3]` AND at least one `mse_*[4]` above threshold.

Tests use `pytest.mark.parametrize` and run in <60 s total on CI (CPU,
small dims, 5 rounds per case).

---

## 8. Error handling

- All wire shape/op_tag/dtype mismatches raise `WireProtocolError`.
- `slalom_verify_safe` returns `inf` on NaN/Inf (already in FFN; lifted
  to shared module).
- Worker `serve_once` ignores empty probe connections.
- Coordinator `close()` sends `MSG_CLOSE` and is idempotent.
- Loopback launcher's worker subprocess: SIGTERM after 1 s grace, then
  SIGKILL after 3 s; only surface non-zero exits other than `-15` in
  stderr (matches FFN behavior).

---

## 9. Out of scope (for follow-up specs)

- Wrapping the real diffusers `ZImageTransformer2DModel`.
- Multi-worker / weight-sharded examples.
- KV cache / autoregressive decoding for the Llama attn example.
- Refactoring `examples/multi_machine_ffn.py` to use `_multi_machine_common.py`.
- Freivalds-based optimization of `q@k^T` and `probs@v` checks (faster
  but harder to read; the FFN-style "redo on coordinator" is the
  pedagogical default).
- A `MULTI_MACHINE_ATTN_ZIMAGE_REPORT.md` analogous to FFN's perf
  report. Generation of that report happens after implementation lands
  and we have measured numbers.

---

## 10. Deliverables

1. `examples/_multi_machine_common.py` (~300 lines)
2. `examples/multi_machine_attn_llama.py` (~500 lines)
3. `examples/multi_machine_attn_zimage.py` (~500 lines)
4. `examples/multi_machine_zimage.py` (~700 lines)
5. `tests/test_multi_machine_examples.py` (parametrized loopback +
   fault tests)
6. This spec, committed to git.

The implementation plan (next step, via `writing-plans`) decomposes
these into ordered tasks with review checkpoints.
