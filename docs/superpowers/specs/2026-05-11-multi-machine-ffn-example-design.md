# Multi-Machine FFN Standalone Example — Design

> A single-file, runnable example showing how the project's multi-machine
> verified inference design (see `docs/MULTI_MACHINE.md`) plays out for one
> SwiGLU feed-forward block. Coordinator + worker as two processes, real
> TCP transport, real SLALOM verification, real bandwidth/latency numbers,
> outputs-only wire.

Status: design, ready for implementation plan.

---

## 1. Goal & Non-Goals

### Goal

Produce `examples/multi_machine_ffn.py`, a ~450-line single-file program
that:

1. **Teaches the multi-machine pattern.** A reader who understands single-
   process verified inference can read this top-to-bottom and grasp the
   chain-of-trust, outputs-only wire, and per-op SLALOM verification
   without touching `verified_core` / `verified_diffusers`.
2. **Produces real measurements.** End-to-end latency, GPU forward time,
   wire bytes (predicted vs measured), CPU SLALOM time, MSE distribution
   — directly comparable to what `tools/distributed_perf_model.py`
   predicts for a SwiGLU MLP at a given `(hidden, inter, dtype)` tuple.
3. **Runs both ways from one codebase.** Default `--role loopback` spawns
   a worker subprocess locally (good for CI). `--role coordinator`
   /`--role worker` split across two real machines.
4. **Demonstrates that verification actually catches misbehavior.**
   `--inject-fault` flag makes the worker provably misbehave; the
   coordinator's SLALOM checks fire.

### Non-Goals

- Not a step toward production. No `VerifyTransport` interface, no gRPC,
  no attestation, no YAML config, no Phase-0 refactor of `verified_core`.
  This is `docs/MULTI_MACHINE.md`'s ideas distilled into a teaching
  artifact; it is explicitly *not* the Phase 1 loopback transport.
- Not a chain of multiple FFNs. Single block, intentionally.
- Not multi-tenant, not data/tensor parallel, not KV-cache aware.
- Not a benchmark for transport. fp16 is the default wire dtype;
  fp8/zstd/RDMA are not in scope.

---

## 2. Architecture

```
┌─ Coordinator process (trusted CPU) ─────────────────────┐
│  • CPU-side weights w1, w2, w3 (loaded from same seed)  │
│  • Random vector s (k=10), s_tilde_{w1,w2,w3} = Wᵀ@s    │
│  • Verifier thread pool (max_workers=2)                 │
│  • Metrics accumulator                                  │
└────────────────────────────┬────────────────────────────┘
                             │ TCP (default 127.0.0.1:9100)
                             ▼
┌─ Worker process (untrusted GPU) ────────────────────────┐
│  • GPU weights w1, w2, w3 (same seed)                   │
│  • No s, no s_tilde, no verification                    │
│  • Optional --inject-fault to misbehave                 │
└─────────────────────────────────────────────────────────┘
```

Two processes. Coordinator drives; worker is a request/response GPU
service. Inputs are never on the wire — coordinator and worker derive
each round's `x` independently from a shared `input_seed`, which IS on
the wire (4 bytes/round, trivial). This stays faithful to the
"outputs-only" principle of `docs/MULTI_MACHINE.md §5.3`: the security
property the seed shortcut preserves is that the worker cannot pick a
forged `(x, y)` pair — both sides derive the same `x` deterministically,
and only `y` is what crosses the network and is checked.

In a real chain (LLM, diffusion), `x_N` is reconstructed from `y_{N-1}`
+ CPU recompute of non-linears; no seed is needed. The seed is the
single-FFN-no-history bootstrap.

---

## 3. Wire Protocol

Custom binary, little-endian, `struct.pack`-based. No external dep.

**Frame**:

```
┌──── 4 bytes ────┬──── 4 bytes ────┬──── payload ────┐
│   msg_type u32  │   body_len u32  │   ...           │
└─────────────────┴─────────────────┴─────────────────┘
```

`msg_type` enum:

```
LOAD_REQ      = 1
LOAD_ACK      = 2
FORWARD_REQ   = 3
ACTIVATION    = 4
FORWARD_DONE  = 5
CLOSE         = 6
```

**Bodies** (`struct.pack` format strings shown):

```
LOAD_REQ      "<IIII"           hidden_dim, inter_dim, weight_seed, dtype_id
LOAD_ACK      "<B"              status (0=ok, 1=fail)
FORWARD_REQ   "<QIII"           request_id, input_seed, batch, seq
ACTIVATION    body packed in 3 segments, concatenated:
                seg1 "<QBBB"      request_id, op_tag, dtype_id, ndim
                seg2 f"<{ndim}I"  shape[0..ndim]
                seg3              raw payload bytes (tensor.numpy().tobytes())
FORWARD_DONE  "<Qd"             request_id, gpu_forward_t_ms
CLOSE         ""                (no body)
```

`op_tag ∈ {W1=1, W3=2, W2=3}`.
`dtype_id ∈ {FP32=1, FP16=2, BF16=3}` (BF16 listed for future, fp16 is
default).
Payload is `tensor.detach().contiguous().cpu().numpy().tobytes()`;
receiver does `np.frombuffer(bytes, dtype).reshape(shape)` then
`torch.from_numpy(...)`.

**Framing read loop** (both sides):

```python
def recv_msg(sock) -> tuple[int, bytes]:
    hdr = recv_exactly(sock, 8)
    msg_type, body_len = struct.unpack("<II", hdr)
    body = recv_exactly(sock, body_len)
    return msg_type, body
```

`recv_exactly` loops on `sock.recv` until `body_len` bytes are in. Raises
`ConnectionResetError` on EOF.

---

## 4. Worker

State machine:

```
boot
  → accept() one client (coordinator)
  → recv LOAD_REQ
  → torch.manual_seed(weight_seed); w1, w2, w3 = make_weights(...) on device
  → send LOAD_ACK(0)
  → loop:
      recv msg
      if CLOSE: break
      if FORWARD_REQ:
          t0 = perf_counter()
          torch.manual_seed(input_seed); x = randn(batch, seq, hidden, device, dtype)
          y1 = w1(x)
          y3 = w3(x)
          gated = silu(y1) * y3
          y2 = w2(gated)
          torch.cuda.synchronize() if device.type == "cuda" else pass
          gpu_t = (perf_counter() - t0) * 1000
          if inject_fault: mutate y1 / y2 / gated per fault flag
          send ACTIVATION(W1, y1)
          send ACTIVATION(W3, y3)
          send ACTIVATION(W2, y2)
          send FORWARD_DONE(request_id, gpu_t)
  → close socket, exit 0
```

`make_weights(hidden, inter, seed, dtype, device)`:

```python
g = torch.Generator(device=device).manual_seed(seed)
w1 = nn.Linear(hidden, inter, bias=False, device=device, dtype=dtype)
w3 = nn.Linear(hidden, inter, bias=False, device=device, dtype=dtype)
w2 = nn.Linear(inter, hidden, bias=False, device=device, dtype=dtype)
for lin in (w1, w3, w2):
    nn.init.normal_(lin.weight, std=0.02, generator=g)
return w1, w2, w3
```

Coordinator runs the **same** function with `device="cpu"`, `dtype=fp32`.
Both sides get bit-identical weights up to dtype downcasting on worker.

**Fault injection** (worker side, post-forward, pre-send):

```python
if inject_fault == "flip_y1":  y1 = -y1
elif inject_fault == "scale_y2": y2 = y2 * 1.01
elif inject_fault == "drop_silu":
    # recompute y2 with broken silo so the on-wire y2 reflects it
    gated_bad = y1 * y3  # missing SiLU
    y2 = w2(gated_bad)
```

`drop_silu` is the chain-of-trust witness: the worker's broken non-linear
never crosses the wire, but its downstream `y2` no longer matches what
the coordinator's CPU `SiLU(y1)*y3 @ s_tilde_w2` produces — the third
SLALOM check fires.

---

## 5. Coordinator

State machine:

```
boot
  → connect to worker
  → send LOAD_REQ
  → recv LOAD_ACK (assert ok)
  → init CPU-side weights from same seed (cpu, fp32)
  → Three independent random vectors (one per layer, because `s` shape depends on
    each layer's output dim):
        s_w1 = randn(inter,  k=10)   s_tilde_w1 = w1.weight.t().fp32 @ s_w1   # (hidden, k)
        s_w3 = randn(inter,  k=10)   s_tilde_w3 = w3.weight.t().fp32 @ s_w3   # (hidden, k)
        s_w2 = randn(hidden, k=10)   s_tilde_w2 = w2.weight.t().fp32 @ s_w2   # (inter,  k)
  → init ThreadPoolExecutor(max_workers=2), metrics list
  → for request_id in 0..rounds-1:
      input_seed = derive(request_id)  # deterministic
      t_send = perf_counter()
      send FORWARD_REQ(request_id, input_seed, batch, seq)

      bytes_recv = 0
      y1_recv, y3_recv, y2_recv = None, None, None
      gpu_t = None
      while not (y1_recv and y3_recv and y2_recv and gpu_t is not None):
          msg_type, body = recv_msg(sock)
          bytes_recv += 8 + len(body)
          if msg_type == ACTIVATION:
              op_tag, tensor = decode_activation(body)
              if op_tag == W1: y1_recv = tensor
              elif op_tag == W3: y3_recv = tensor
              elif op_tag == W2: y2_recv = tensor
          elif msg_type == FORWARD_DONE:
              gpu_t = decode_done(body).gpu_forward_t_ms

      # Reproduce x on CPU
      torch.manual_seed(input_seed); x_cpu = randn(batch, seq, hidden, dtype=fp32)

      # Submit SLALOM in parallel (w1, w3 independent of each other)
      f_w1 = pool.submit(slalom_verify, x_cpu, y1_recv.to(fp32), s_w1, s_tilde_w1)
      f_w3 = pool.submit(slalom_verify, x_cpu, y3_recv.to(fp32), s_w3, s_tilde_w3)
      mse_w1 = f_w1.result()
      mse_w3 = f_w3.result()
      # w2 depends on y1_recv + y3_recv being on CPU (they already are by this point)
      gated_cpu = F.silu(y1_recv.to(fp32)) * y3_recv.to(fp32)
      f_w2 = pool.submit(slalom_verify, gated_cpu, y2_recv.to(fp32), s_w2, s_tilde_w2)
      mse_w2 = f_w2.result()

      metrics.append(RoundMetrics(...))
  → send CLOSE, close socket
  → print summary
```

The pool gives true parallelism for the `w1`/`w3` SLALOM matmuls (numpy
and torch release the GIL inside large matmul). `w2` runs sequentially
after them because it needs `gated_cpu` which depends on both.

---

## 6. SLALOM (inlined, ~20 lines)

```python
def make_s(hidden: int, k: int = 10, dtype=torch.float32) -> torch.Tensor:
    # Random projection vectors, shape (hidden, k)
    return torch.randn(hidden, k, dtype=dtype)

def precompute_s_tilde(weight: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # weight is nn.Linear.weight, shape (out, in). We compute (in, k) = W.T @ s.
    # The matched check is y @ s == x @ (W.T @ s) for y = x @ W.T.
    return weight.detach().to(s.dtype).t().contiguous() @ s

def slalom_verify(
    x: torch.Tensor,         # (B, S, in)  fp32
    y: torch.Tensor,         # (B, S, out) fp32
    s: torch.Tensor,         # (out, k)    fp32
    s_tilde: torch.Tensor,   # (in, k)     fp32
) -> float:
    lhs = y @ s              # (B, S, k)
    rhs = x @ s_tilde        # (B, S, k)
    return ((lhs - rhs) ** 2).mean().item()
```

Threshold default `1e-3` (fp16 wire round-trip is the dominant error
floor; SwiGLU-7B-shaped tests in `verified_diffusers` already use a
similar value).

---

## 7. Metrics

```python
@dataclass
class RoundMetrics:
    request_id: int
    coord_send_t: float          # ms, time-on-wire from FORWARD_REQ to first ACTIVATION
    gpu_forward_t: float         # ms, worker-reported
    wire_recv_t: float           # ms, recv of 3 ACTIVATION + FORWARD_DONE
    cpu_verify_t: float          # ms, parallel max(w1,w3) + sequential w2
    end_to_end_t: float          # ms, FORWARD_REQ send → SLALOM w2 done
    bytes_sent: int
    bytes_recv: int
    bytes_recv_predicted: int
    mse_w1: float
    mse_w3: float
    mse_w2: float
    ok: bool
```

`bytes_recv_predicted` (assuming 3D activation tensors, ndim=3):
```
# 3 ACTIVATION frames + 1 FORWARD_DONE frame
frame_hdr        = 8                      # msg_type + body_len
activation_body  = 11 + 4 * 3             # Q+B+B+B + 3 shape u32 = 23
forward_done_body = 16                    # Q + d
overhead = 3 * (frame_hdr + activation_body) + (frame_hdr + forward_done_body)
         = 3 * 31 + 24 = 117

y1_bytes = batch * seq * inter  * dtype_size
y3_bytes = batch * seq * inter  * dtype_size
y2_bytes = batch * seq * hidden * dtype_size
predicted = overhead + y1_bytes + y3_bytes + y2_bytes
```

**Summary report** (stdout after N rounds):

```
=== Multi-Machine FFN Example: 100 rounds ===

Config:
  FFN:      SwiGLU  hidden=4096  inter=11008  dtype=fp16  batch×seq=1×512
  Wire:     fp16    transport=TCP/127.0.0.1:9100
  Verify:   SLALOM  k=10  threshold=1e-3
  Worker:   cuda:0

End-to-end (ms):
  p50   <...>   p95   <...>   mean   <...>
  GPU forward       <...>
  Wire RTT          <...>     (≈ <X> MB/round → <Y> Gbps effective)
  CPU SLALOM        <...>     (parallel max(w1,w3) + w2)

Wire bytes (per round):
  Predicted     <X>.X MB
  Measured      <X>.X MB     (Δ=<...>% framing overhead)

Verification:
  rounds passed     N / N
  mse_w1 p95        <...>
  mse_w3 p95        <...>
  mse_w2 p95        <...>
```

`--verbose` prints one line per round.

---

## 8. Failure Handling

| Failure | Detection | Response |
|---|---|---|
| SLALOM mse > threshold | `mse > config.threshold` | `ok=False` on round, log `tag=Wx mse=...`, continue rounds (educational value of seeing distribution) |
| NaN/Inf in `y` | `torch.isfinite(y).all()` | Same + `[NAN]` tag in log |
| Worker socket EOF / reset | `recv_exactly` raises | Coordinator raises, loopback mode dumps worker stderr, exit code 1 |
| Coord socket timeout | `socket.settimeout(30)` | Same as above |
| Shape/dtype mismatch | Validate against expected `(batch, seq, hidden)` or `(batch, seq, inter)` | Raise `WireProtocolError(expected=..., got=...)` |
| LOAD_ACK timeout (5 s) | Coordinator timeout on first recv | Exit with "worker did not boot" |

`--inject-fault` matrix for self-test:

| Fault | Which check catches it | Expected mse |
|---|---|---|
| `flip_y1` | SLALOM `w1` | huge |
| `scale_y2` | SLALOM `w2` | huge |
| `drop_silu` | SLALOM `w2` (via chain) | huge |
| `none` | — | < threshold |

---

## 9. CLI

```
--role           {loopback, coordinator, worker}     default: loopback
--bind           HOST:PORT                           worker only, default: 127.0.0.1:9100
--worker-host    HOST                                coordinator only
--worker-port    INT                                 default: 9100
--rounds         INT                                 default: 100
--hidden         INT                                 default: 4096
--inter          INT                                 default: 11008
--batch          INT                                 default: 1
--seq            INT                                 default: 512
--wire-dtype     {fp16, fp32}                        default: fp16
--weight-seed    INT                                 default: 0xC0FFEE
--threshold      FLOAT                               default: 1e-3
--device         {cuda:0, cuda:1, ..., cpu}          worker, default: cuda:0
--inject-fault   {none, flip_y1, scale_y2, drop_silu} worker, default: none
--verbose        flag, per-round metrics
```

---

## 10. Loopback Launcher

`--role loopback` (default) does:

```python
def launch_loopback(args):
    port = pick_free_port()
    worker_cmd = [
        sys.executable, __file__,
        "--role", "worker", "--bind", f"127.0.0.1:{port}",
        # forward shape/seed/dtype/fault flags
    ]
    worker_proc = subprocess.Popen(worker_cmd, stderr=subprocess.PIPE)
    try:
        wait_for_port_listen(port, timeout=10)
        run_coordinator(args, host="127.0.0.1", port=port)
    finally:
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_proc.kill()
        if worker_proc.returncode != 0:
            sys.stderr.write(worker_proc.stderr.read().decode())
```

Port discovery via `socket.bind(("", 0))` then close; pass the
allocated port to the worker subprocess. Avoids hardcoding 9100 in CI.

---

## 11. Testing

`tests/test_multi_machine_ffn_example.py`. Each test starts a worker
subprocess on a random port and runs a coordinator in-process (so we
can assert on metrics directly).

| Test | What it checks |
|---|---|
| `test_loopback_passes_clean` | 5 rounds, all `ok=True`, all mse < threshold |
| `test_inject_flip_y1_caught` | `--inject-fault flip_y1` → 100% rounds fail, `mse_w1 > threshold` |
| `test_inject_scale_y2_caught` | Same for `scale_y2` → `mse_w2 > threshold` |
| `test_inject_drop_silu_caught` | Same for `drop_silu` → `mse_w2 > threshold`; proves SiLU is implicitly verified |
| `test_wire_bytes_predicted_matches` | `abs(measured - predicted) / predicted < 0.01` |
| `test_close_message_shuts_worker` | After CLOSE, worker exits 0 within 2 s |
| `test_wire_dtype_fp32_also_works` | `--wire-dtype fp32` passes clean |

Each test ≤ 2 seconds wall-clock. Total suite < 15 s.

---

## 12. File Manifest

```
examples/multi_machine_ffn.py            ~450 lines  new
tests/test_multi_machine_ffn_example.py  ~180 lines  new
```

No edits to existing files. Dependencies: `torch`, `numpy`, Python
stdlib (`socket`, `struct`, `subprocess`, `argparse`,
`concurrent.futures`, `dataclasses`, `time`).

---

## 13. Relationship to `docs/MULTI_MACHINE.md`

What the example **does** implement from the production design:

- Outputs-only wire (§5.3): yes, only `y1`/`y3`/`y2` cross
- Coordinator owns `s` / `s_tilde` (§8): yes
- Implicit element-wise verification via downstream SLALOM (§5.3 table,
  bottom row): yes, `drop_silu` test proves it
- SLALOM mse threshold + `fail_on_error` semantics (§9): yes
- Single-process backward-compat (§11 Phase 0): N/A — example is not
  meant to be migratable to production runtime

What the example **does not** implement (intentionally):

- `VerifyTransport` interface (§6.1): the example codes its TCP layer
  inline; a real port to production would extract the same shape.
- Heartbeat / session lifecycle (§5.4): no — the example uses one
  connection per process lifetime; if either side dies the run aborts.
- TLS + attestation (§6.1, §10): no — loopback.
- gRPC / protobuf (§6.1): no — raw TCP suffices for one message family.
- Multi-worker fleet, data parallel routing (§5.2, §11 Phase 3): no — one
  worker, one model, one FFN.
- YAML config schema (§10): no — CLI flags only.

The example is the smallest faithful demonstration of the wire shape and
verification flow — readers should be able to map every line to a
§-numbered concept in the design doc.

---

## 14. Open Questions

1. **Default `--inter`.** SwiGLU MLP in real models has inter ≈ 2.7×
   hidden (Llama-7B uses 4096→11008). Default to `--hidden 4096
   --inter 11008` for familiarity. Document `--help` so a user
   benchmarking against `tools/distributed_perf_model.py` can pass the
   `(hidden, inter)` pair from whichever config they care about.
2. **`s_tilde` precompute timing.** For 4096×11008 fp32 the precompute
   is ~50 ms — negligible vs 100 forward rounds, but mention in the
   summary report (`startup_t` line).
3. **Verbose mode formatting.** Per-round lines should be one line each,
   ≤ 100 chars, so CI logs stay readable. Plan: `r=<id> e2e=<ms> gpu=<ms>
   wire=<ms> verify=<ms> mse=<w1>/<w3>/<w2> ok=<T/F>`.
