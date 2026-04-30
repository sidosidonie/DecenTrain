# Multi-Machine Verified Inference

> Design for extending the current single-process verified inference stack
> (`verified_core` + `verified_llm` + `verified_diffusers`) into a
> multi-machine deployment that matches the topology already modelled in
> `tools/distributed_perf_model.py` and described by `configs/*.yaml`:
> **1 CPU TEE coordinator + N remote GPU workers**, connected by a TCP/RDMA
> network instead of PCIe.

This is a design document. No runtime code is changed by it.

---

## 1. Goal

Today the verified runtime assumes GPU and CPU live in the same process and
share a PCIe bus. The performance model in
`tools/distributed_perf_model.py` already simulates a different layout —
many small / mid-tier GPUs on separate hosts feeding one trusted CPU
through 1–100 GbE — and configs like `configs/home_8gpu_7b_fleet.yaml`
already describe that fleet. We want the actual runtime to support the
same topology so:

1. One TEE host (TDX-capable Xeon, EPYC SEV-SNP, etc.) can verify several
   independent GPU workers running in parallel.
2. GPU workers can be commodity boxes (4090 / 5090 / single-GPU tower)
   sitting at home or in a small cluster, with no TEE of their own.
3. Each worker stays simple: load model, run forward pass, ship the
   activations the TEE asks for. All trust, key material and verification
   logic stay on the TEE.
4. The existing single-process path keeps working — multi-machine is an
   additional deployment mode, not a replacement.

A second hard constraint, which shapes everything below:

> **The wire carries matmul / matmul-like outputs only.** Inputs are
> never shipped from worker to coordinator — they are reconstructed on
> the coordinator from a CPU-side chain of trust (previous verified
> output + CPU recompute of non-linear ops). See §5.3.

Non-goals (explicitly out of scope for this document):

- Multi-tenant isolation between users sharing one TEE.
- Cross-worker tensor parallel / pipeline parallel for one big model
  (we focus on the simulator's *standalone* strategy first).
- New verification algorithms — SLALOM and Freivalds are unchanged.

---

## 2. Current Architecture (single process)

```
┌──────────────────────────────────────────────────────────────────┐
│                        Python process                            │
│                                                                  │
│   ┌──────────────────────┐        ┌──────────────────────────┐   │
│   │   GPU (untrusted)    │        │   CPU (TEE assumed)      │   │
│   │                      │        │                          │   │
│   │  forward(x) → y      │        │   VerifyRuntime          │   │
│   │  Q@K^T, P@V          │ PCIe   │     copy_stream          │   │
│   │                      │ D2H    │     ThreadPoolExecutor   │   │
│   │                      │ ──────►│     SLALOM / Freivalds   │   │
│   │                      │        │     cpu_state (chain)    │   │
│   └──────────────────────┘        └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

Key APIs (all in-process):

| API | Where | What it does |
|---|---|---|
| `VerifyRuntime.submit_linear_preprocessed(tag, x_gpu, y_gpu, s, s_tilde)` | `verified_core/runtime.py` | Async D2H + SLALOM check |
| `VerifyRuntime.submit_matmul(tag, a, b, c)` | same | Async D2H + Freivalds check |
| `VerifyRuntime.submit_elementwise(tag, in, out, op_name)` | same | D2H + CPU recompute |
| `VerifyRuntime.cpu_state_set_d2h / cpu_state_get` | same | Chain-of-trust state between layers |
| `VerifyRuntime.flush()` | same | Wait for all pending checks |

The interesting property: **`submit_*` is fire-and-forget from the caller's
perspective.** All transport (PCIe), all compute (SLALOM/Freivalds), all
state (`_cpu_state*`) are hidden behind the runtime. That gives us the
seam to cut at.

---

## 3. Target Architecture (multi-machine)

```
                    ┌────────────────────────────────────────────┐
                    │            TEE host (trusted)              │
                    │                                            │
                    │   ┌────────────────────────────────────┐   │
   per-worker       │   │  Coordinator process               │   │
   RPC channels     │   │   - per-worker VerifyRuntime       │   │
   ┌──────────────► │   │   - SLALOM / Freivalds workers     │   │
   │                │   │   - cpu_state per worker           │   │
   │                │   │   - aggregated profiler            │   │
   │                │   └────────────────────────────────────┘   │
   │                └────────────────────────────────────────────┘
   │                          ▲           ▲           ▲
   │ network                  │           │           │
   │ (TCP / RDMA)             │           │           │
   ▼                          │           │           │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ GPU worker 1 │    │ GPU worker 2 │    │ GPU worker 3 │    │ GPU worker N │
│ 1× RTX 5090  │    │ 1× RTX 4090  │    │ 1× RTX 4090  │    │ 1× ...       │
│ Qwen2.5-7B   │    │ Llama-3.2-1B │    │ Z-Image      │    │ ...          │
│ untrusted    │    │ untrusted    │    │ untrusted    │    │ untrusted    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

Two process types:

- **Coordinator** (one, runs on the TEE host).
  Owns: SLALOM precompute (`s`, `s_tilde`), per-worker `VerifyRuntime`
  instances (without the CUDA streams), threadpool, profiler, error log.
  Does **not** own GPU.
- **GPU worker** (one per machine / per GPU).
  Owns: full model weights, CUDA context, forward pass.
  Does **not** own verification keys, does **not** verify anything.
  Treated as untrusted — the only trust property we want is
  "either the worker returns activations consistent with its declared
  model, or the coordinator catches it."

---

## 4. Mapping from the Simulator to Runtime

The simulator already exposes the parameters that matter. Here's how each
maps onto the multi-machine runtime:

| Simulator concept | Runtime concept |
|---|---|
| `cluster.strategy: standalone` | One worker process, model fits on its GPU(s) |
| `cluster.strategy: pipeline_parallel` | Single worker process, multi-GPU within that worker (existing PP) |
| `cluster.strategy: tensor_parallel` | Single worker process, multi-GPU within (existing TP) |
| `cluster.strategy: data_parallel` | N replica workers behind a load balancer (later) |
| `cluster.gpu_type` / `count` | Worker host hardware; coordinator only sees "a worker" |
| `cluster.model` | Worker loads the model; coordinator loads weights only for SLALOM precompute |
| `cpu.preset` / `cpu.tee` | Coordinator host; TEE attestation happens here |
| `network.preset` | Wire between coordinator and worker. Drives transport choice |
| `verify.use_slalom` / `freivalds_k` / `verify_every_n` / `compression` | Coordinator's `VerifyConfig`. `compression: fp16` selects fp16 wire format |
| `verify.tee_overhead_pct` | Modeled — runtime just accepts the slowdown from running inside TDX/SNP |

Configs need almost no schema change: a `cluster` already implicitly
describes one worker. We only need to add an optional `endpoint` (host:port,
attestation policy) per cluster. See §10.

---

## 5. Component Design

### 5.1 GPU worker

Minimal surface. The worker is intentionally close to "stock PyTorch
inference + a network shim". A worker:

1. On boot, loads the model exactly like today's `create_llm_model(...)`
   or `VerifiedZImagePipeline.from_pretrained(...)`, but with verification
   **disabled locally** (`VerifyConfig(enabled=False)`).
2. Replaces the `VerifyLinearModule` / `VerifyMatmul` / `VerifyRuntime`
   surface with a `RemoteVerifyClient`. The wrappers' `forward` still
   does GPU compute exactly as before; the only change is that
   `submit_linear_preprocessed`, `submit_matmul`, `submit_elementwise`
   instead of doing local D2H + ThreadPool, push tensors out over the
   wire.
3. Owns no `s`, `s_tilde`, no random seeds for verification.
4. Exposes a control RPC (`Load`, `Forward`, `GenerateStream`, `Close`,
   `GetProfile`) used by the coordinator to drive inference.
5. Has no privileged state. If it goes away mid-stream the coordinator
   just declares the request failed.

The worker can be implemented as a thin subclass / strategy that swaps
out `VerifyRuntime`. Concretely we add a `VerifyTransport` interface
(see §6) and pass it into `VerifyRuntime`. Local mode uses the existing
`InProcessTransport`; multi-machine mode uses `RemoteWorkerTransport`.

### 5.2 Coordinator (TEE host)

The coordinator looks a lot like today's process **with the GPU work
removed**:

- Boots, parses the YAML config, opens N RPC channels (one per
  `cluster`).
- For each cluster: tells the worker to load the named model, then
  fetches the weight metadata it needs. **Weights themselves do not
  need to leave the TEE host if the operator pre-stages them on the
  coordinator** (see §9 / §10 for attestation), but in a relaxed
  threat model the coordinator can also re-derive `s_tilde` from a
  local copy of the weights. Either way, `s` and `s_tilde` live on
  the coordinator only.
- Allocates a `VerifyRuntime` per worker (no CUDA streams; the
  ThreadPool and SLALOM/Freivalds stay).
- Routes user-level requests (chat completion, image generation) to
  the right worker, and routes the worker's activation snapshots back
  into the right `VerifyRuntime`.
- On `flush()` failure, tags the offending worker, drains its queue,
  and surfaces a `RuntimeError` exactly like single-process mode.

### 5.3 Outputs-only wire shape (chain-of-trust)

Plain SLALOM (Tramer & Boneh 2019) takes both `x` and `y` and checks
`y @ s == x @ s_tilde`. The naive port of that to a network would D2H
both `x` and `y` per linear layer, doubling wire load for no gain. The
project already side-steps this in `verified_llm/attn_layer.py`:

> `GPU runs the complete forward pass (all matmuls + non-linears).
> CPU asynchronously verifies — only matmul OUTPUTS are D2H copied.
> All inputs come from the CPU chain (maintained across layers via
> runtime cpu_state). Non-linear operations (layernorm, q_norm, RoPE,
> softmax, sigmoid) are recomputed on CPU from chain data.`

In multi-machine mode this becomes a hard rule, not a per-layer choice.
The coordinator reconstructs every input on its own:

```
                   coordinator                            worker (GPU)

  cpu_state["layer_input_L"]  ───┐                  forward pass on GPU
                                 │
            RMSNorm on CPU       │
                                 ▼
                          x_L (CPU-side)
                                 │
                                 │   ◄── y_L (q/k/v/o or gate/up/down) ──── network
                                 ▼
                  SLALOM check  y_L @ s  vs  x_L @ s_tilde
                                 │
                  CPU recompute: bias + reshape + RoPE + softmax + ...
                                 │
                                 ▼
                  cpu_state["layer_input_{L+1}"]    (= input to next layer)
```

**What crosses the wire (per verified op):**

| GPU op | Worker → coordinator | Source of inputs at coordinator |
|---|---|---|
| Linear `y = x @ W^T` (Q/K/V/O, gate/up/down) | `y` only | `x` from `cpu_state` (previous layer output + CPU norms) |
| Dynamic matmul `C = A @ B` (Q@K^T, P@V) | `C` only | `A`, `B` from CPU chain (RoPE-applied Q/K, recomputed P) |
| Element-wise (softmax, SiLU) | **nothing** | CPU recomputes the op from chain `IN` and uses the CPU result as ground truth for downstream checks |

The element-wise output never leaves the worker because its correctness
is implicit in the next matmul's check: if the worker's softmax differs
from the CPU's softmax, then the worker's `P @ V` will not match the
coordinator's `P @ V`, and the next Freivalds check fails.

**Wire savings (vs. inputs+outputs naive port):**

| Model | Per-layer attn savings | Per-layer MLP savings | Per-token decode savings |
|---|---|---|---|
| Qwen3.5-9B (H=4096, I=12288) | drops `x` once per attn (16 KB fp16) | drops `x` once per MLP (8 KB fp16) | ~24 KB × layers ≈ 1 MB/token |
| Llama-3-70B (H=8192, I=28672) | 32 KB / attn | 16 KB / MLP | ~3 MB/token |
| Z-Image (seq>>1) | dominant input savings (`x` is full token grid) | similar | proportionally larger |

The simulator's `estimate_transfer_bytes` currently sums `x + outputs`
per layer (see `tools/distributed_perf_model.py`). It needs an
`outputs_only=True` path; numbers above are the deltas.

**Threat-model gain.** With `x` on the wire, an adversarial worker
could choose any `(x, y)` pair satisfying `y @ s == x @ s_tilde` and
pass the check. Outputs-only forces the worker to match `y` to a `x`
it does not control: the coordinator-side `x` is derived from the
previous verified output plus deterministic non-linear ops, which the
worker cannot influence without first failing an earlier check.

**Bootstrap.** The chain has to start somewhere. For LLMs, the
coordinator owns the embedding lookup table (small — 100–500 MB for
0.5–9B models, fits trivially in TEE memory) and runs the embedding
+ first RMSNorm itself, seeding `cpu_state["layer_input_0"]` locally.
For diffusion the equivalent is the patchifier and time/text embedding
on coordinator — also cheap. The worker never needs to send any
"primary input" tensor to the coordinator.

**Implication: legacy auto-submit path.** The current
`VerifyRuntime.submit_linear_preprocessed(x_gpu, y_gpu, s, s_tilde)`
signature D2Hs both `x` and `y`. It is used by `verified_llm` (still
on the chain path; the duplicated `x` D2H is wasted) and by
`verified_diffusers/zimage/layers.py:VerifyLinearModule.forward()`
(auto-submit path, no chain). For multi-machine to work, every layer
wrapper must move to the chain pattern. Phase 0 of the migration
(§11) covers this refactor.

### 5.4 Connection / lifecycle

- Coordinator → worker: persistent bi-directional stream per worker
  (HTTP/2 + gRPC, or QUIC, see §6).
- A "session" = one (coordinator, worker, model) tuple. Created at
  boot, reused across many forward passes. SLALOM `s_tilde` is bound
  to a session; rotating `s` invalidates the session.
- Worker can host one model at a time (matches `standalone`); we leave
  multi-tenant slicing for later.
- Heartbeat every N seconds. If a worker stops responding, the
  coordinator marks the session failed and (configurable) re-issues
  pending requests on a different worker if a replica exists.

---

## 6. Wire Protocol

Two layers: a **transport** (bytes in / bytes out) and a **message
format** on top of it.

### 6.1 Transport

| Network bracket | Recommended transport | Rationale |
|---|---|---|
| 1–10 GbE home | gRPC over HTTP/2 + TLS | Easy, mature, fp16 compression already a knob |
| 25 GbE small cluster | gRPC + zstd, or QUIC | Same code path, more headroom |
| 100 GbE+ datacenter | RDMA write + small control gRPC | Only meaningful if SLALOM stops being CPU-bound |
| Same NUMA / same host | UNIX domain sockets | Useful for tests and "fake distributed" mode |

We start with a single transport: **gRPC over HTTP/2**. It gives streaming,
TLS, attestation hooks, and language-agnostic clients for free. We hide
it behind the `VerifyTransport` interface so we can swap to RDMA later
without touching `verified_llm` / `verified_diffusers`.

`VerifyTransport` interface (sketch):

```python
class VerifyTransport(Protocol):
    def submit_linear(self, tag: str,
                      x: TensorRef, y: TensorRef,
                      s_id: str) -> Future[VerifyResult]: ...
    def submit_matmul(self, tag: str,
                      a: TensorRef, b: TensorRef, c: TensorRef) -> Future[VerifyResult]: ...
    def submit_elementwise(self, tag: str,
                           x: TensorRef, y: TensorRef,
                           op_name: str) -> Future[VerifyResult]: ...
    def flush(self) -> None: ...
```

`InProcessTransport` keeps current behaviour. `RemoteWorkerTransport` is
the new one.

### 6.2 Message format

Three message families.

**Control** (small, infrequent):

```
LoadModel { model_name, dtype, hf_revision, attn_impl }   → LoadResult { layer_meta }
Forward    { request_id, input_ids, attention_mask, kv_id } → ForwardAck { stream_id }
GenerateStream { request_id, max_new_tokens, ... }        → token stream
Close      { session_id }                                 → CloseAck
GetProfile { session_id }                                 → ProfileBlob (CSV/JSON)
```

**Activation upload** (high volume, per verified op). Per the
chain-of-trust principle in §5.3, **only outputs cross the wire**:

```
Activation {
  session_id      : u64
  request_id      : u64
  layer_idx       : u32
  op              : enum { LINEAR, MATMUL }      # element-wise outputs are not shipped
  tag             : string             # e.g. "block.3.attn.q_proj"
  s_id            : string             # SLALOM key id (for LINEAR)
  output          : TensorChunk        # y for LINEAR, c for MATMUL — exactly one chunk
  d2h_done_ts     : u64                # nanoseconds since session start
}

TensorChunk {
  role            : enum { Y, C }      # output of linear, output of dynamic matmul
  shape           : repeated u32
  dtype           : enum { FP32, FP16, BF16 }
  layout          : enum { ROW_MAJOR, ROW_MAJOR_PADDED }
  payload         : bytes              # raw, optionally fp16 / fp8 compressed
}
```

Notes:

- No `X`, `A`, `B`, `IN` roles. Inputs are reconstructed coordinator-
  side from `cpu_state` + CPU recompute (see §5.3).
- No `ELEMENTWISE_*` ops. The worker's softmax / SiLU output is
  discarded; the coordinator uses its own CPU recompute and its
  correctness is tested implicitly by the next matmul's check.
- Each `Activation` is **exactly one TensorChunk**. Simpler framing,
  smaller per-message overhead, and matches the runtime's existing
  one-D2H-per-op shape.

Compression knob = simulator's `verify.compression`. fp16 halves the
wire cost (already validated by the simulator). fp8 and zstd are
follow-ups.

**Verification result** (small, per op): pushed back from the coordinator
to the worker only when `fail_on_error=False` and the worker wants to
react (rare). With `fail_on_error=True` (default) the coordinator just
logs and the worker never hears about the failure until the next
heartbeat tells it the session is dead.

### 6.3 Backpressure

A worker can outrun the coordinator's CPU. We enforce a per-worker
in-flight cap (`max_pending_ops_per_worker`) implemented as a credit
system on the gRPC stream. When credits run out, the worker pauses its
GPU forward at the next `submit_*` boundary. This matches the
single-process behaviour where a saturated `ThreadPoolExecutor` queue
naturally throttles forward progress.

---

## 7. Verification Flow Walkthrough

Suppose a user posts a chat request that hits **worker #2** (4090 running
Qwen2.5-7B). At layer 3, per token, the sequence is:

```
[worker#2 GPU]                              [TEE coordinator]

(coordinator already holds                  cpu_state["layer_input_3"]
 because layer 2's chain wrote it           = previous verified output
 at the end of layer 2)                       + residual + RMSNorm input
                                            x_3 = RMSNorm(layer_input_3)
                                                  on CPU

x_3.q = GPU(x_3 @ W_q^T)
  ── Activation { L=3, LINEAR, tag=q_proj,
                  output=Y(fp16),
                  s_id="qwen7b@2:L3.q" } ──► SLALOM:  y@s == x_3@s_tilde
                                              (x_3 is CPU-derived,
                                               worker did not send it)

x_3.k, x_3.v same way ─────────────────►   SLALOM checks queued

(CPU chain: bias, reshape, q_norm,
 RoPE — uses cos/sin already shared
 once at session start)
                                            Q_rot, K_rot, V on CPU

scores = GPU(Q_rot @ K_rot^T)
  ── Activation { L=3, MATMUL, tag=qk,
                  output=C(fp16) } ───────► Freivalds:
                                              C(fp16) vs (Q_rot @ K_rot)
                                              (A,B from chain — not shipped)

(GPU runs softmax on its scores, but        CPU recomputes:
 its result never crosses the wire)         P_cpu = softmax(C_received)

attn_out = GPU(P_gpu @ V)
  ── Activation { L=3, MATMUL, tag=pv,
                  output=C(fp16) } ───────► Freivalds:
                                              attn_out vs (P_cpu @ V)
                                              ─ if worker's softmax was
                                                wrong, P_gpu ≠ P_cpu and
                                                this check fails

o_raw = GPU(attn_out @ W_o^T)
  ── Activation { L=3, LINEAR, tag=o_proj,
                  output=Y(fp16) } ───────► SLALOM check
                                            CPU does residual + post-norm
                                            cpu_state["layer_input_4"] set

… MLP gate / up / down: 3 LINEAR ops, output-only, same pattern …

(end of forward)
worker emits ForwardDone { request_id }
                                            flush(); raise on any failure
                                            → return generated token
```

Three things to notice:

1. **Wire is roughly half** of the inputs+outputs naive port. The
   simulator's `estimate_transfer_bytes` needs an `outputs_only` path
   (today it sums `x + y`) — see §5.3 table for the deltas.
2. **No element-wise traffic.** The worker's softmax output is dropped
   on the floor; only its downstream effect (in `P @ V`) is checked.
3. **Chain-of-trust state stays inside the coordinator.** The worker
   never sees `cpu_state`, never sees `s` or `s_tilde`, never sees
   `cos / sin` after session start. That is what makes outputs-only
   safe — the worker cannot pick a forged `(x, y)` pair to satisfy
   the SLALOM equation, because it does not see the `x` the
   coordinator will compare against.

---

## 8. SLALOM Partitioning

SLALOM precompute is the security-critical step:

- `weight_t` lives on **both** sides (worker uses it on GPU, coordinator
  uses a copy in CPU memory only to derive `s_tilde`).
- `s` is a fresh random vector generated by the coordinator. Never sent
  to the worker. Rotated per session.
- `s_tilde = (W^T cast through gpu_dtype) @ s` is computed by the
  coordinator and kept in CPU memory only.

The wire only ever carries the matmul / matmul-like **outputs** (`y`,
`c`) and tags — see §5.3 for why inputs are not shipped. The worker
cannot pre-compute `y @ s` because it doesn't have `s`, and it cannot
pick `y` to match a fake `x` because the coordinator's `x` comes from
the chain and is not exposed to the worker. Combined, this preserves
the SLALOM threat model from the original paper across the network
boundary while halving the bytes on the wire.

If the operator does **not** want to ship weights to the coordinator
(e.g. hosted GPU service distributing closed weights), there are two
fallbacks:

1. Run with `verify.use_slalom: false`. Coordinator falls back to
   standard Freivalds, which only needs the weight matrix on the
   verifier *if it doesn't already have `s_tilde`*. This costs the
   100×–200× speedup SLALOM gives us. The simulator quantifies this
   exactly in `run_slalom_vs_freivalds()`.
2. Have the worker prove `s_tilde` correctness via remote attestation
   plus a one-shot SLALOM precompute that runs *inside the worker's
   own TEE-GPU enclave* (H100 confidential computing). Out of scope
   for v1.

---

## 9. Failure Handling and Security

| Failure mode | Detection | Response |
|---|---|---|
| Worker returns wrong matmul | SLALOM / Freivalds MSE > threshold | Mark session failed, raise on `flush()`, log to profiler with `ok=False` |
| Worker returns NaN / Inf | Existing fallback path in `runtime.py` (`math.isfinite(loss)`) | Same as today — masked / recomputed |
| Worker crashes mid-stream | Heartbeat / RPC error | Drain pending futures with errors, surface to caller |
| Coordinator crashes | Worker heartbeat timeout | Worker drops session, holds no privileged state |
| Network partition | RPC deadline | Same as worker crash |
| Slow worker (CPU verify keeps up but worker GPU stalls) | Token tok/s drops | Existing profiler captures it; not a security event |
| TEE attestation failure on coordinator boot | Quote verification | Refuse to start; never load weights |

Important: **a verification failure is never silent.** Multi-machine mode
must keep `fail_on_error=True` semantics from `VerifyConfig`. Any worker
producing a failed check is quarantined.

We do **not** broadcast failures to other workers; each worker is
independent in `standalone` strategy.

---

## 10. Configuration

We extend `configs/*.yaml` minimally. Existing fields keep their
meaning. New fields are optional (when missing, runtime falls back to
single-process / loopback mode for backward compatibility):

```yaml
cpu:
  preset: Xeon-8480+
  tee: tdx
  endpoint:                       # NEW (optional)
    bind: 0.0.0.0:9000
    tls_cert_pem: /etc/tee/cert.pem
    tls_key_pem:  /etc/tee/key.pem
    attestation:
      mode: tdx_quote             # tdx_quote | sev_snp_report | none
      policy_path: ./policy.yaml

clusters:
  - name: 4090-7b-a
    gpu_type: RTX-4090
    count: 1
    strategy: standalone
    model: Qwen2.5-7B
    endpoint:                     # NEW (optional)
      host: 192.168.1.21
      port: 9100
      tls_ca_pem: /etc/tee/ca.pem
      worker_id: w-4090-a

network:
  preset: 10GbE
  inflight_credits: 64            # NEW (optional, default = 32)

verify:
  use_slalom: true
  freivalds_k: 10
  verify_every_n: 1
  compression: fp16               # already supported by sim; wire format here
```

Single-process mode = no `endpoint` blocks. The coordinator launches an
in-process worker via `InProcessTransport`. This keeps every existing
test green.

---

## 11. Migration Plan (phased)

We ship in vertical slices so each phase is independently usable.

**Phase 0 — chain-only auto-submit + transport seam, no behaviour change**

- Make outputs-only the only verification path (§5.3):
  - `verified_diffusers/zimage/layers.py:VerifyLinearModule.forward()`
    auto-submit path is removed or rewritten on top of the chain
    pattern already used in `verified_llm/attn_layer.py` /
    `mlp_layer.py`. Z-Image attention / MLP / transformer-block
    wrappers gain explicit `cpu_state` keys for entry/exit.
  - Add a coordinator-side bootstrap helper (`build_layer0_input`)
    that owns the embedding lookup / patchifier so the chain has a
    seed. Move that compute off the GPU worker for both LLM and
    diffusion paths.
  - `VerifyRuntime.submit_linear_preprocessed` API simplifies to
    `submit_linear_output_only(tag, y_gpu, x_cpu_key, s_id)`; the
    legacy `(x_gpu, y_gpu)` overload is deprecated and removed in
    the same commit train.
- Extract `VerifyTransport` interface from `VerifyRuntime`. Default
  binding stays in-process.
- Move CUDA stream + ThreadPool + profiler from `VerifyRuntime` into a
  local `InProcessTransport`. `submit_*` becomes a thin dispatcher.
- All existing tests keep passing without touching them, *except* the
  Z-Image tests that exercised the auto-submit path — those switch to
  the chain path and assert the same end-to-end behaviour.

**Phase 1 — loopback transport**

- Add `RemoteWorkerTransport` and a worker process that talks to itself
  over a UNIX socket (or localhost gRPC). Coordinator and worker run on
  the same machine, same Python interpreter via `multiprocessing`.
- New end-to-end test: `tests/test_loopback_transport.py`. Same model,
  same prompts, same outputs as in-process mode. Verifies wire schema,
  serialization, fp16 compression round-trip, error propagation.

**Phase 2 — single remote worker**

- Real gRPC server in the worker, real client in the coordinator.
- One config: `configs/home_4090_5090.yaml` (already in repo) becomes
  runnable as two boxes.
- Add `tools/run_worker.py` and `tools/run_coordinator.py` entrypoints.

**Phase 3 — fleet (N workers, 1 TEE)**

- Coordinator multiplexes across N sessions. One `VerifyRuntime` per
  session, sharing the threadpool and profiler.
- Run `configs/home_8gpu_7b_fleet.yaml` end-to-end on 1 TDX host + 8
  consumer GPU boxes.
- Add aggregate dashboards: per-worker tok/s, per-worker fail count,
  TEE CPU saturation.

**Phase 4 — production hardening**

- TLS + remote attestation policy (TDX quote / SEV-SNP report) gating
  worker registration.
- Backpressure tuning, retry / replica routing for `data_parallel`.
- Optional RDMA transport behind the same `VerifyTransport` interface
  (only worth it after benchmarking shows network is the bottleneck).

Each phase keeps the simulator (`tools/distributed_perf_model.py`,
`tools/perf_analyze.py`) as the source of truth for *what we expect*;
the new code's job is to land within ~20% of the simulator's prediction
for the same `(model, gpu, cpu, network)` tuple.

---

## 12. What we explicitly punt on

- **Cross-worker tensor parallel.** A single 70B model striped across
  multiple GPU workers via the network is interesting but very different
  in flow control and bandwidth profile. Today's `cluster_70b_tp.yaml`
  models that *within one worker host*. Treat cross-host TP as a v2
  effort.
- **Confidential GPU.** H100 / B200 confidential computing would let us
  trust the worker GPU and skip verification. Orthogonal to this design.
- **Streaming the KV cache.** Today the KV cache lives on the worker.
  We keep it that way; only activations cross the wire. KV resharding
  for multi-replica `data_parallel` is its own design.
- **Multi-tenant scheduling.** One coordinator serves one logical tenant
  in v1.

---

## 13. Open questions

1. **Where do weights live for SLALOM precompute?** Coordinator-only is
   simplest; coordinator + worker both is more flexible but doubles
   storage. Lean coordinator-only for now.
2. **Random vector reuse.** Today `_RandomVecCache` reuses `r` per
   `(n,k,dtype)`. Across the wire, do we tie `s` to the session and
   rotate per session? Probably yes — cheap to regenerate, removes a
   replay class.
3. **Bootstrap embedding cost.** Coordinator running embedding lookup +
   first RMSNorm itself is ~1–5 ms/token on the TEE CPU for ≤9B models
   (vocab × hidden small). Negligible at TDX overhead, but we should
   measure for 70B-vocab models to confirm it does not become the
   bottleneck.
4. **Z-Image chain refactor.** The Z-Image stack today auto-submits
   `(x, y)` per linear instead of using `cpu_state`. Phase 0 of §11
   has to land that refactor before multi-machine works for diffusion.
   Risk: small — `verified_llm/attn_layer.py` is the proven template.
5. **Profiler aggregation.** Per-worker CSV vs single coordinator CSV?
   Coordinator-side is the single source of truth; worker-side is
   advisory only. Match `verified_core/profiler.py`'s schema so existing
   tooling keeps working.
6. **fp8 wire format.** Worth it once SLALOM is verified to be CPU-bound
   on the coordinator at 100 GbE; before that, fp16 is enough.
7. **Auth.** mTLS is the obvious default; do we *require* TDX/SNP
   attestation on the coordinator, or accept "operator-trusted host"
   as a deployment knob?

---

## 14. Summary

The simulator already encodes the topology, the bandwidth costs and the
SLALOM speedup. The runtime today already encapsulates verification
behind `VerifyRuntime.submit_*`. The work is mostly plumbing: introduce
a `VerifyTransport` seam, build a gRPC-backed remote transport, split the
process into coordinator + worker, and wire YAML configs end-to-end.
Every phase keeps the existing single-process tests green, which keeps
risk bounded.
