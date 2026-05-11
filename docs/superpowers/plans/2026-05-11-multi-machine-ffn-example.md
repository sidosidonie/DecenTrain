# Multi-Machine FFN Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, runnable example (`examples/multi_machine_ffn.py`) that demonstrates the project's multi-machine verified-inference design for one SwiGLU FFN block: coordinator + worker over TCP, outputs-only wire, real SLALOM verification, full perf breakdown.

**Architecture:** Single Python file, two processes (one coordinator, one worker), raw TCP socket transport with `struct`-packed frames. Coordinator owns SLALOM keys, weights (CPU fp32), and metrics. Worker owns GPU weights and runs forward. Inputs derived deterministically on both sides from a shared seed so only outputs cross the wire. Parallel SLALOM verification via `ThreadPoolExecutor(max_workers=2)`.

**Tech Stack:** Python 3.10+, `torch`, `numpy`, stdlib (`socket`, `struct`, `subprocess`, `argparse`, `concurrent.futures`, `dataclasses`, `json`, `time`, `enum`). No third-party deps beyond what the project already uses.

**Spec:** `docs/superpowers/specs/2026-05-11-multi-machine-ffn-example-design.md` — read this first if any detail below is unclear.

---

## File Structure

```
examples/multi_machine_ffn.py            new  ~500 lines, single file with sections:
                                              CONSTANTS | WEIGHTS | SLALOM | WIRE |
                                              WORKER    | COORD   | METRICS | MAIN
tests/test_multi_machine_ffn_example.py  new  ~200 lines, fast functional tests
tests/test_multi_machine_ffn_perf.py     new  ~180 lines, pytest -m perf, gated
```

No existing files are modified.

The single-file layout keeps the example readable as one document. The two test files split fast functional tests (always run in CI) from perf tests (manual / gated).

---

## Conventions used throughout

- Run tests from repo root: `pytest -q tests/test_multi_machine_ffn_example.py`
- Run perf tests: `pytest -q tests/test_multi_machine_ffn_perf.py -m perf`
- Use a temporary Python file path for incremental testing only when explicitly noted; otherwise tests target the real example file.
- Each task ends with **one commit**. Commit message format: `feat(ffn-example): <subject>` for code; `test(ffn-example): <subject>` for tests.
- All imports go at the top of the file. Add as needed each task — the engineer should consolidate when noticed (but not in the middle of a task).

---

## Task 1: Skeleton — file header, constants, dtype mapping

**Files:**
- Create: `examples/multi_machine_ffn.py`
- Test: `tests/test_multi_machine_ffn_example.py` (create with a single import-only test)

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_machine_ffn_example.py`:

```python
"""Functional tests for examples/multi_machine_ffn.py."""
from __future__ import annotations

import importlib.util
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_ffn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mmffn", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_and_exposes_constants():
    m = _load_module()
    assert m.MSG_LOAD_REQ == 1
    assert m.MSG_LOAD_ACK == 2
    assert m.MSG_FORWARD_REQ == 3
    assert m.MSG_ACTIVATION == 4
    assert m.MSG_FORWARD_DONE == 5
    assert m.MSG_CLOSE == 6
    assert m.OP_W1 == 1
    assert m.OP_W3 == 2
    assert m.OP_W2 == 3
    assert m.DTYPE_FP32 == 1
    assert m.DTYPE_FP16 == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_module_imports_and_exposes_constants -v`
Expected: FAIL with `FileNotFoundError` or `ModuleNotFoundError`.

- [ ] **Step 3: Write the skeleton file**

Create `examples/multi_machine_ffn.py`:

```python
"""Standalone multi-machine FFN example.

Coordinator + worker over TCP. Outputs-only wire. Real SLALOM
verification of a single SwiGLU MLP. See
docs/superpowers/specs/2026-05-11-multi-machine-ffn-example-design.md
for the full design.

Run:
    python examples/multi_machine_ffn.py                       # loopback
    python examples/multi_machine_ffn.py --role worker         # remote
    python examples/multi_machine_ffn.py --role coordinator \\
        --worker-host 192.168.1.21
"""
from __future__ import annotations

# ── Imports ─────────────────────────────────────────────────────────
import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ───────────────────────────────────────────────────────
# Message types
MSG_LOAD_REQ = 1
MSG_LOAD_ACK = 2
MSG_FORWARD_REQ = 3
MSG_ACTIVATION = 4
MSG_FORWARD_DONE = 5
MSG_CLOSE = 6

# Activation op tags
OP_W1 = 1
OP_W3 = 2
OP_W2 = 3

# Wire dtypes
DTYPE_FP32 = 1
DTYPE_FP16 = 2

_TORCH_DTYPE = {DTYPE_FP32: torch.float32, DTYPE_FP16: torch.float16}
_NUMPY_DTYPE = {DTYPE_FP32: np.float32, DTYPE_FP16: np.float16}
_DTYPE_SIZE = {DTYPE_FP32: 4, DTYPE_FP16: 2}
_DTYPE_NAME = {DTYPE_FP32: "fp32", DTYPE_FP16: "fp16"}
_NAME_TO_DTYPE = {v: k for k, v in _DTYPE_NAME.items()}

# SLALOM
SLALOM_K = 10
S_GENERATOR_SEED = 0xDEADBEEF  # fixed so tests are deterministic


def main() -> int:  # pragma: no cover - filled in last task
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_module_imports_and_exposes_constants -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): skeleton with wire constants and dtype mapping"
```

---

## Task 2: Deterministic weight initialization

**Files:**
- Modify: `examples/multi_machine_ffn.py` (add `make_weights`)
- Modify: `tests/test_multi_machine_ffn_example.py` (add weight tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
import torch


def test_make_weights_deterministic():
    m = _load_module()
    w1_a, w2_a, w3_a = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    w1_b, w2_b, w3_b = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    assert torch.equal(w1_a.weight, w1_b.weight)
    assert torch.equal(w2_a.weight, w2_b.weight)
    assert torch.equal(w3_a.weight, w3_b.weight)


def test_make_weights_shapes():
    m = _load_module()
    w1, w2, w3 = m.make_weights(hidden=8, inter=16, seed=0, dtype=torch.float32, device="cpu")
    assert w1.weight.shape == (16, 8)
    assert w2.weight.shape == (8, 16)
    assert w3.weight.shape == (16, 8)
    assert w1.bias is None and w2.bias is None and w3.bias is None


def test_make_weights_cross_device_dtype_consistency():
    """CPU fp32 and (would-be) GPU fp16 produce same values up to dtype cast."""
    m = _load_module()
    w1_cpu, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    w1_half, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float16, device="cpu")
    # fp32 -> fp16 -> fp32 should match what we got from direct fp16
    assert torch.allclose(w1_cpu.weight.to(torch.float16).to(torch.float32),
                          w1_half.weight.to(torch.float32))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k make_weights -v`
Expected: FAIL with `AttributeError: module 'mmffn' has no attribute 'make_weights'`.

- [ ] **Step 3: Implement `make_weights`**

In `examples/multi_machine_ffn.py`, after the constants block, add:

```python
# ── Weight initialization ───────────────────────────────────────────
def make_weights(
    hidden: int,
    inter: int,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Build SwiGLU weights (w1, w2, w3) deterministic across CPU/GPU.

    We always sample on the CPU with std=0.02, then `.to(device, dtype)`,
    so the coordinator (CPU fp32) and worker (GPU fp16) get bit-identical
    weights up to the dtype downcast.

    Returns (w1, w2, w3) — same order as zimage/mlp.py for consistency.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _lin(in_dim: int, out_dim: int) -> nn.Linear:
        m = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            m.weight.normal_(0.0, 0.02, generator=gen)
        return m.to(device=device, dtype=dtype)

    w1 = _lin(hidden, inter)
    w3 = _lin(hidden, inter)
    w2 = _lin(inter, hidden)
    return w1, w2, w3
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k make_weights -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): deterministic make_weights across CPU/GPU"
```

---

## Task 3: SLALOM verifier (math core)

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_slalom_verify_passes_for_correct_matmul():
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    mse = m.slalom_verify(x, y, s, s_tilde)
    assert mse < 1e-6, f"expected ~0 mse, got {mse}"


def test_slalom_verify_catches_corrupted_y():
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    y_bad = y * 1.01  # 1% scale → should be way above threshold
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    mse = m.slalom_verify(x, y_bad, s, s_tilde)
    assert mse > 1e-3, f"expected large mse, got {mse}"


def test_precompute_s_tilde_shape():
    m = _load_module()
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    assert s_tilde.shape == (8, 10)  # (in, k)


def test_make_s_deterministic_with_seed():
    m = _load_module()
    a = m.make_s(out_dim=16, k=10, seed=7)
    b = m.make_s(out_dim=16, k=10, seed=7)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k slalom -v`
Expected: FAIL with `AttributeError: module 'mmffn' has no attribute 'slalom_verify'`.

- [ ] **Step 3: Implement SLALOM helpers**

In `examples/multi_machine_ffn.py`, after `make_weights`, add:

```python
# ── SLALOM ──────────────────────────────────────────────────────────
def make_s(out_dim: int, k: int, seed: int) -> torch.Tensor:
    """Random projection vector. Shape (out_dim, k), fp32 on CPU."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(out_dim, k, dtype=torch.float32, generator=gen)


def precompute_s_tilde(weight: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """For nn.Linear y = x @ weight.T, compute s_tilde = weight.T @ s.

    weight shape: (out, in).  s shape: (out, k).  Returns (in, k) fp32.
    """
    w = weight.detach().to(torch.float32).t().contiguous()  # (in, out)
    return w @ s.to(torch.float32)  # (in, k)


def slalom_verify(
    x: torch.Tensor,        # (..., in)  fp32
    y: torch.Tensor,        # (..., out) fp32 — received from worker
    s: torch.Tensor,        # (out, k)   fp32
    s_tilde: torch.Tensor,  # (in, k)    fp32
) -> float:
    """Return mean-squared-error between y@s and x@s_tilde.

    A correct (x, y) pair gives ~0 mse. Any forged y that doesn't match
    `x @ W.T` will diverge with high probability across k=10 projections.
    """
    lhs = y.to(torch.float32) @ s
    rhs = x.to(torch.float32) @ s_tilde
    return ((lhs - rhs) ** 2).mean().item()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "slalom or precompute or make_s" -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): SLALOM verify + precompute helpers"
```

---

## Task 4: Wire primitives — frame header send/recv

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
import socket as _socket


def test_send_recv_msg_roundtrip():
    m = _load_module()
    a, b = _socket.socketpair()
    try:
        m.send_msg(a, m.MSG_LOAD_REQ, b"hello")
        msg_type, body = m.recv_msg(b)
        assert msg_type == m.MSG_LOAD_REQ
        assert body == b"hello"
    finally:
        a.close()
        b.close()


def test_send_recv_msg_empty_body():
    m = _load_module()
    a, b = _socket.socketpair()
    try:
        m.send_msg(a, m.MSG_CLOSE, b"")
        msg_type, body = m.recv_msg(b)
        assert msg_type == m.MSG_CLOSE
        assert body == b""
    finally:
        a.close()
        b.close()


def test_recv_exactly_raises_on_eof():
    import pytest
    m = _load_module()
    a, b = _socket.socketpair()
    a.close()  # immediately
    try:
        with pytest.raises(ConnectionError):
            m.recv_exactly(b, 10)
    finally:
        b.close()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "send_recv or recv_exactly" -v`
Expected: FAIL — missing functions.

- [ ] **Step 3: Implement wire primitives**

In `examples/multi_machine_ffn.py`, after SLALOM section, add:

```python
# ── Wire primitives ─────────────────────────────────────────────────
class WireProtocolError(RuntimeError):
    pass


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock or raise ConnectionError on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"unexpected EOF after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, msg_type: int, body: bytes) -> int:
    """Send a framed message. Returns total bytes written (header+body)."""
    header = struct.pack("<II", msg_type, len(body))
    sock.sendall(header + body)
    return len(header) + len(body)


def recv_msg(sock: socket.socket) -> tuple[int, bytes]:
    """Read one framed message. Returns (msg_type, body)."""
    header = recv_exactly(sock, 8)
    msg_type, body_len = struct.unpack("<II", header)
    body = recv_exactly(sock, body_len) if body_len else b""
    return msg_type, body
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "send_recv or recv_exactly" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): framed TCP send/recv primitives"
```

---

## Task 5: Wire pack/unpack for each message type

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_pack_unpack_load_req():
    m = _load_module()
    body = m.pack_load_req(hidden=4096, inter=11008, weight_seed=0xC0FFEE, dtype_id=m.DTYPE_FP16)
    out = m.unpack_load_req(body)
    assert out == {"hidden": 4096, "inter": 11008, "weight_seed": 0xC0FFEE, "dtype_id": m.DTYPE_FP16}


def test_pack_unpack_forward_req():
    m = _load_module()
    body = m.pack_forward_req(request_id=7, input_seed=99, batch=1, seq=512)
    out = m.unpack_forward_req(body)
    assert out == {"request_id": 7, "input_seed": 99, "batch": 1, "seq": 512}


def test_pack_unpack_activation_roundtrip():
    m = _load_module()
    t = torch.randn(2, 4, 8, dtype=torch.float16)
    body = m.pack_activation(request_id=3, op_tag=m.OP_W1, tensor=t, wire_dtype_id=m.DTYPE_FP16)
    out = m.unpack_activation(body)
    assert out["request_id"] == 3
    assert out["op_tag"] == m.OP_W1
    assert out["tensor"].shape == (2, 4, 8)
    assert out["tensor"].dtype == torch.float16
    assert torch.allclose(out["tensor"], t)


def test_pack_unpack_forward_done():
    m = _load_module()
    body = m.pack_forward_done(request_id=5, gpu_forward_t_ms=12.345)
    out = m.unpack_forward_done(body)
    assert out["request_id"] == 5
    assert abs(out["gpu_forward_t_ms"] - 12.345) < 1e-9
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "pack_unpack" -v`
Expected: FAIL — missing functions.

- [ ] **Step 3: Implement pack/unpack**

In `examples/multi_machine_ffn.py`, after wire primitives, add:

```python
# ── Wire message bodies ─────────────────────────────────────────────
def pack_load_req(hidden: int, inter: int, weight_seed: int, dtype_id: int) -> bytes:
    return struct.pack("<IIII", hidden, inter, weight_seed, dtype_id)


def unpack_load_req(body: bytes) -> dict:
    hidden, inter, weight_seed, dtype_id = struct.unpack("<IIII", body)
    return {"hidden": hidden, "inter": inter, "weight_seed": weight_seed, "dtype_id": dtype_id}


def pack_load_ack(status: int) -> bytes:
    return struct.pack("<B", status)


def unpack_load_ack(body: bytes) -> dict:
    (status,) = struct.unpack("<B", body)
    return {"status": status}


def pack_forward_req(request_id: int, input_seed: int, batch: int, seq: int) -> bytes:
    return struct.pack("<QIII", request_id, input_seed, batch, seq)


def unpack_forward_req(body: bytes) -> dict:
    request_id, input_seed, batch, seq = struct.unpack("<QIII", body)
    return {"request_id": request_id, "input_seed": input_seed, "batch": batch, "seq": seq}


def pack_activation(
    request_id: int, op_tag: int, tensor: torch.Tensor, wire_dtype_id: int
) -> bytes:
    np_dtype = _NUMPY_DTYPE[wire_dtype_id]
    torch_dtype = _TORCH_DTYPE[wire_dtype_id]
    t = tensor.detach().to(torch_dtype).contiguous().cpu()
    payload = t.numpy().astype(np_dtype, copy=False).tobytes()
    ndim = t.ndim
    shape_bytes = struct.pack(f"<{ndim}I", *t.shape)
    header = struct.pack("<QBBB", request_id, op_tag, wire_dtype_id, ndim)
    return header + shape_bytes + payload


def unpack_activation(body: bytes) -> dict:
    request_id, op_tag, dtype_id, ndim = struct.unpack_from("<QBBB", body, 0)
    off = struct.calcsize("<QBBB")
    shape = struct.unpack_from(f"<{ndim}I", body, off)
    off += 4 * ndim
    payload = body[off:]
    np_dtype = _NUMPY_DTYPE[dtype_id]
    torch_dtype = _TORCH_DTYPE[dtype_id]
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape)
    # numpy buffer is read-only; .copy() before torch.from_numpy
    tensor = torch.from_numpy(arr.copy()).to(torch_dtype)
    return {"request_id": request_id, "op_tag": op_tag, "dtype_id": dtype_id, "tensor": tensor}


def pack_forward_done(request_id: int, gpu_forward_t_ms: float) -> bytes:
    return struct.pack("<Qd", request_id, gpu_forward_t_ms)


def unpack_forward_done(body: bytes) -> dict:
    request_id, gpu_forward_t_ms = struct.unpack("<Qd", body)
    return {"request_id": request_id, "gpu_forward_t_ms": gpu_forward_t_ms}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "pack_unpack" -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): pack/unpack for all wire message bodies"
```

---

## Task 6: Worker class — load only

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing test**

We need a helper to spawn workers as subprocesses. Append to `tests/test_multi_machine_ffn_example.py`:

```python
import subprocess
import time as _time
import sys as _sys


def _pick_free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> None:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            with _socket.create_connection((host, port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            _time.sleep(0.05)
    raise TimeoutError(f"worker port {port} did not open within {timeout}s")


class _WorkerProc:
    """Context manager that spawns a worker subprocess on a free port."""

    def __init__(self, **cli):
        self.cli = cli
        self.port = _pick_free_port()
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        cmd = [_sys.executable, str(EXAMPLE_PATH),
               "--role", "worker",
               "--bind", f"127.0.0.1:{self.port}",
               "--device", "cpu"]  # tests default to CPU for portability
        for k, v in self.cli.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            _wait_port(self.port)
        except TimeoutError:
            self.proc.terminate()
            out, err = self.proc.communicate(timeout=2)
            raise RuntimeError(
                f"worker did not start. stdout={out!r} stderr={err!r}"
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def test_worker_handles_load_and_close():
    """Bring up worker, send LOAD_REQ, get LOAD_ACK, send CLOSE, worker exits."""
    m = _load_module()
    with _WorkerProc() as wp:
        with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
            m.send_msg(sock, m.MSG_LOAD_REQ,
                       m.pack_load_req(hidden=8, inter=16,
                                        weight_seed=42, dtype_id=m.DTYPE_FP32))
            mt, body = m.recv_msg(sock)
            assert mt == m.MSG_LOAD_ACK
            assert m.unpack_load_ack(body)["status"] == 0
            m.send_msg(sock, m.MSG_CLOSE, b"")
        # worker should exit cleanly within 3s
        rc = wp.proc.wait(timeout=3)
        assert rc == 0, f"worker exited with {rc}"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_worker_handles_load_and_close -v`
Expected: FAIL — worker CLI not implemented.

- [ ] **Step 3: Implement minimal Worker class**

In `examples/multi_machine_ffn.py`, after wire bodies section, add:

```python
# ── Worker ──────────────────────────────────────────────────────────
class Worker:
    """Untrusted GPU host. Serves one coordinator at a time."""

    def __init__(self, bind_host: str, bind_port: int, device: str,
                 inject_fault: str = "none"):
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.device = torch.device(device)
        self.inject_fault = inject_fault
        self.w1: Optional[nn.Linear] = None
        self.w2: Optional[nn.Linear] = None
        self.w3: Optional[nn.Linear] = None
        self.hidden = 0
        self.inter = 0
        self.wire_dtype_id = DTYPE_FP16
        self.compute_dtype: torch.dtype = torch.float16

    def serve_once(self) -> None:
        """Accept one client, serve until CLOSE or disconnect."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.bind_port))
        srv.listen(1)
        try:
            sock, _addr = srv.accept()
        finally:
            srv.close()
        try:
            self._serve_session(sock)
        finally:
            sock.close()

    def _serve_session(self, sock: socket.socket) -> None:
        while True:
            try:
                msg_type, body = recv_msg(sock)
            except (ConnectionError, OSError):
                return
            if msg_type == MSG_LOAD_REQ:
                fields = unpack_load_req(body)
                self._handle_load(sock, fields)
            elif msg_type == MSG_CLOSE:
                return
            elif msg_type == MSG_FORWARD_REQ:
                fields = unpack_forward_req(body)
                self._handle_forward(sock, fields)
            else:
                raise WireProtocolError(f"unexpected msg_type {msg_type} on worker")

    def _handle_load(self, sock: socket.socket, fields: dict) -> None:
        self.hidden = fields["hidden"]
        self.inter = fields["inter"]
        self.wire_dtype_id = fields["dtype_id"]
        self.compute_dtype = _TORCH_DTYPE[self.wire_dtype_id]
        self.w1, self.w2, self.w3 = make_weights(
            self.hidden, self.inter, fields["weight_seed"],
            dtype=self.compute_dtype, device=self.device,
        )
        send_msg(sock, MSG_LOAD_ACK, pack_load_ack(0))

    def _handle_forward(self, sock: socket.socket, fields: dict) -> None:
        raise NotImplementedError("forward implemented in Task 7")
```

- [ ] **Step 4: Wire up worker role in main()**

Replace the `main()` stub at the bottom of `examples/multi_machine_ffn.py`:

```python
# ── Main (incremental — full CLI in last task) ──────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["loopback", "worker", "coordinator"],
                   default="loopback")
    p.add_argument("--bind", default="127.0.0.1:9100")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject-fault", default="none",
                   choices=["none", "flip_y1", "scale_y2", "drop_silu"])
    args, _unknown = p.parse_known_args()

    if args.role == "worker":
        host, port_s = args.bind.split(":")
        Worker(host, int(port_s), args.device, args.inject_fault).serve_once()
        return 0
    raise NotImplementedError(f"role={args.role} not yet implemented")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_worker_handles_load_and_close -v`
Expected: PASS within 5 seconds.

- [ ] **Step 6: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): worker handles LOAD_REQ and CLOSE"
```

---

## Task 7: Worker forward pass (clean, no fault injection)

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def _reproduce_input(seed: int, batch: int, seq: int, hidden: int,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, seq, hidden, dtype=torch.float32, generator=gen).to(dtype)


def test_worker_forward_produces_correct_outputs():
    m = _load_module()
    with _WorkerProc() as wp:
        with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
            m.send_msg(sock, m.MSG_LOAD_REQ,
                       m.pack_load_req(hidden=8, inter=16,
                                        weight_seed=42, dtype_id=m.DTYPE_FP32))
            mt, _ = m.recv_msg(sock)
            assert mt == m.MSG_LOAD_ACK

            m.send_msg(sock, m.MSG_FORWARD_REQ,
                       m.pack_forward_req(request_id=1, input_seed=99,
                                           batch=1, seq=4))
            # Expect 3 ACTIVATION + 1 FORWARD_DONE
            received = []
            for _ in range(4):
                mt, body = m.recv_msg(sock)
                received.append((mt, body))
            m.send_msg(sock, m.MSG_CLOSE, b"")

        acts = [b for (t, b) in received if t == m.MSG_ACTIVATION]
        dones = [b for (t, b) in received if t == m.MSG_FORWARD_DONE]
        assert len(acts) == 3 and len(dones) == 1

        # Reproduce expected outputs from same seed + same weights on CPU
        w1, w2, w3 = m.make_weights(hidden=8, inter=16, seed=42,
                                     dtype=torch.float32, device="cpu")
        x = _reproduce_input(seed=99, batch=1, seq=4, hidden=8)
        y1_expected = w1(x)
        y3_expected = w3(x)
        y2_expected = w2(F.silu(y1_expected) * y3_expected)

        by_tag = {}
        for body in acts:
            d = m.unpack_activation(body)
            by_tag[d["op_tag"]] = d["tensor"]
        assert torch.allclose(by_tag[m.OP_W1], y1_expected, atol=1e-4)
        assert torch.allclose(by_tag[m.OP_W3], y3_expected, atol=1e-4)
        assert torch.allclose(by_tag[m.OP_W2], y2_expected, atol=1e-4)

        done = m.unpack_forward_done(dones[0])
        assert done["request_id"] == 1
        assert done["gpu_forward_t_ms"] > 0
```

Also add `import torch.nn.functional as F` at the top of the test file if not already there.

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_worker_forward_produces_correct_outputs -v`
Expected: FAIL with `NotImplementedError: forward implemented in Task 7`.

- [ ] **Step 3: Implement forward in Worker**

In `examples/multi_machine_ffn.py`, replace `Worker._handle_forward` with:

```python
    def _handle_forward(self, sock: socket.socket, fields: dict) -> None:
        request_id = fields["request_id"]
        input_seed = fields["input_seed"]
        batch = fields["batch"]
        seq = fields["seq"]

        # Deterministic input reproduction. Same recipe runs on coordinator.
        gen = torch.Generator(device="cpu").manual_seed(input_seed)
        x_cpu = torch.randn(batch, seq, self.hidden,
                             dtype=torch.float32, generator=gen)
        x = x_cpu.to(device=self.device, dtype=self.compute_dtype)

        t0 = time.perf_counter()
        y1 = self.w1(x)
        y3 = self.w3(x)
        gated = F.silu(y1) * y3
        y2 = self.w2(gated)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_t_ms = (time.perf_counter() - t0) * 1000.0

        y1, y3, y2 = self._apply_fault(y1, y3, y2, gated)

        send_msg(sock, MSG_ACTIVATION,
                 pack_activation(request_id, OP_W1, y1, self.wire_dtype_id))
        send_msg(sock, MSG_ACTIVATION,
                 pack_activation(request_id, OP_W3, y3, self.wire_dtype_id))
        send_msg(sock, MSG_ACTIVATION,
                 pack_activation(request_id, OP_W2, y2, self.wire_dtype_id))
        send_msg(sock, MSG_FORWARD_DONE,
                 pack_forward_done(request_id, gpu_t_ms))

    def _apply_fault(self, y1, y3, y2, gated):
        """No-op for inject_fault='none'; overridden in Task 8."""
        return y1, y3, y2
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_worker_forward_produces_correct_outputs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): worker clean forward returns 3 activations + done"
```

---

## Task 8: Fault injection in worker

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def _run_one_round_collect_acts(wp, *, hidden, inter, input_seed=99,
                                 batch=1, seq=4):
    m = _load_module()
    with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
        m.send_msg(sock, m.MSG_LOAD_REQ,
                   m.pack_load_req(hidden=hidden, inter=inter,
                                    weight_seed=42, dtype_id=m.DTYPE_FP32))
        m.recv_msg(sock)
        m.send_msg(sock, m.MSG_FORWARD_REQ,
                   m.pack_forward_req(request_id=1, input_seed=input_seed,
                                       batch=batch, seq=seq))
        acts = {}
        while len(acts) < 3:
            mt, body = m.recv_msg(sock)
            if mt == m.MSG_ACTIVATION:
                d = m.unpack_activation(body)
                acts[d["op_tag"]] = d["tensor"]
        m.recv_msg(sock)  # FORWARD_DONE
        m.send_msg(sock, m.MSG_CLOSE, b"")
    return acts


def test_fault_flip_y1_negates_first_output():
    m = _load_module()
    with _WorkerProc(inject_fault="none") as clean, \
         _WorkerProc(inject_fault="flip_y1") as bad:
        clean_acts = _run_one_round_collect_acts(clean, hidden=8, inter=16)
        bad_acts = _run_one_round_collect_acts(bad, hidden=8, inter=16)
        assert torch.allclose(bad_acts[m.OP_W1], -clean_acts[m.OP_W1])
        # w3 unaffected
        assert torch.allclose(bad_acts[m.OP_W3], clean_acts[m.OP_W3])


def test_fault_scale_y2_scales_third_output():
    m = _load_module()
    with _WorkerProc(inject_fault="none") as clean, \
         _WorkerProc(inject_fault="scale_y2") as bad:
        clean_acts = _run_one_round_collect_acts(clean, hidden=8, inter=16)
        bad_acts = _run_one_round_collect_acts(bad, hidden=8, inter=16)
        assert torch.allclose(bad_acts[m.OP_W2], clean_acts[m.OP_W2] * 1.01,
                              atol=1e-4)


def test_fault_drop_silu_changes_y2_via_chain():
    m = _load_module()
    with _WorkerProc(inject_fault="none") as clean, \
         _WorkerProc(inject_fault="drop_silu") as bad:
        clean_acts = _run_one_round_collect_acts(clean, hidden=8, inter=16)
        bad_acts = _run_one_round_collect_acts(bad, hidden=8, inter=16)
        # y1 and y3 same as clean (linear projections unaffected)
        assert torch.allclose(bad_acts[m.OP_W1], clean_acts[m.OP_W1])
        assert torch.allclose(bad_acts[m.OP_W3], clean_acts[m.OP_W3])
        # but y2 differs because gated was computed without SiLU
        assert not torch.allclose(bad_acts[m.OP_W2], clean_acts[m.OP_W2])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k fault -v`
Expected: FAIL — faults not implemented (currently always no-op).

- [ ] **Step 3: Implement fault injection**

In `examples/multi_machine_ffn.py`, replace `Worker._apply_fault`:

```python
    def _apply_fault(self, y1, y3, y2, gated):
        if self.inject_fault == "none":
            return y1, y3, y2
        if self.inject_fault == "flip_y1":
            return -y1, y3, y2
        if self.inject_fault == "scale_y2":
            return y1, y3, y2 * 1.01
        if self.inject_fault == "drop_silu":
            # Recompute y2 with broken non-linear; ship that as y2.
            gated_bad = y1 * y3  # missing SiLU
            y2_bad = self.w2(gated_bad)
            return y1, y3, y2_bad
        raise ValueError(f"unknown inject_fault: {self.inject_fault}")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k fault -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): worker fault injection (flip_y1/scale_y2/drop_silu)"
```

---

## Task 9: Coordinator — connect, load, build SLALOM state

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_coordinator_connect_loads_weights_and_builds_slalom():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            # Coordinator owns CPU-side weights and SLALOM state
            assert coord.w1.weight.shape == (16, 8)
            assert coord.s_w1.shape == (16, m.SLALOM_K)
            assert coord.s_tilde_w1.shape == (8, m.SLALOM_K)
            assert coord.s_w2.shape == (8, m.SLALOM_K)
            assert coord.s_tilde_w2.shape == (16, m.SLALOM_K)
        finally:
            coord.close()
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_coordinator_connect_loads_weights_and_builds_slalom -v`
Expected: FAIL — Coordinator class missing.

- [ ] **Step 3: Add FFNConfig and Coordinator skeleton**

In `examples/multi_machine_ffn.py`, after Worker class, add:

```python
# ── Coordinator ─────────────────────────────────────────────────────
@dataclass
class FFNConfig:
    hidden: int
    inter: int
    batch: int
    seq: int
    wire_dtype: int = DTYPE_FP16
    weight_seed: int = 0xC0FFEE


class Coordinator:
    """Trusted host. Owns SLALOM state and verifies every linear output."""

    def __init__(self, host: str, port: int, config: FFNConfig,
                 threshold: float = 1e-3, k: int = SLALOM_K):
        self.host = host
        self.port = port
        self.config = config
        self.threshold = threshold
        self.k = k
        # CPU-side fp32 copy of weights (used for s_tilde precompute)
        self.w1, self.w2, self.w3 = make_weights(
            config.hidden, config.inter, config.weight_seed,
            dtype=torch.float32, device="cpu",
        )
        # Three independent SLALOM projection vectors (one per layer).
        # Fixed seed so verification is reproducible across runs.
        self.s_w1 = make_s(config.inter, k, seed=S_GENERATOR_SEED + 1)
        self.s_w3 = make_s(config.inter, k, seed=S_GENERATOR_SEED + 2)
        self.s_w2 = make_s(config.hidden, k, seed=S_GENERATOR_SEED + 3)
        self.s_tilde_w1 = precompute_s_tilde(self.w1.weight, self.s_w1)
        self.s_tilde_w3 = precompute_s_tilde(self.w3.weight, self.s_w3)
        self.s_tilde_w2 = precompute_s_tilde(self.w2.weight, self.s_w2)

        self.sock: Optional[socket.socket] = None
        self.pool = ThreadPoolExecutor(max_workers=2)

    def connect_and_load(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        send_msg(self.sock, MSG_LOAD_REQ,
                 pack_load_req(self.config.hidden, self.config.inter,
                                self.config.weight_seed, self.config.wire_dtype))
        mt, body = recv_msg(self.sock)
        if mt != MSG_LOAD_ACK:
            raise WireProtocolError(f"expected LOAD_ACK, got {mt}")
        if unpack_load_ack(body)["status"] != 0:
            raise RuntimeError("worker LOAD failed")

    def close(self) -> None:
        if self.sock is not None:
            try:
                send_msg(self.sock, MSG_CLOSE, b"")
            except OSError:
                pass
            self.sock.close()
            self.sock = None
        self.pool.shutdown(wait=False)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_coordinator_connect_loads_weights_and_builds_slalom -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): coordinator connect/load and SLALOM state"
```

---

## Task 10: Coordinator — single round with parallel SLALOM

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_coordinator_run_round_clean_passes_verification():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            rm = coord.run_round(request_id=1, input_seed=99)
            assert rm.ok is True
            assert rm.mse_w1 < 1e-3
            assert rm.mse_w3 < 1e-3
            assert rm.mse_w2 < 1e-3
        finally:
            coord.close()


def test_coordinator_run_round_with_flip_y1_caught():
    m = _load_module()
    with _WorkerProc(inject_fault="flip_y1") as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            rm = coord.run_round(request_id=1, input_seed=99)
            assert rm.ok is False
            assert rm.mse_w1 > 1e-3
        finally:
            coord.close()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k run_round -v`
Expected: FAIL — `Coordinator.run_round` not implemented, `RoundMetrics` not defined.

- [ ] **Step 3: Add RoundMetrics dataclass**

In `examples/multi_machine_ffn.py`, just before the Coordinator class, add:

```python
# ── Metrics ─────────────────────────────────────────────────────────
@dataclass
class RoundMetrics:
    request_id: int
    # Phase timings (ms)
    coord_send_t: float = 0.0
    gpu_forward_t: float = 0.0
    wire_recv_t: float = 0.0
    cpu_verify_t: float = 0.0
    cpu_verify_w1_t: float = 0.0
    cpu_verify_w3_t: float = 0.0
    cpu_verify_w2_t: float = 0.0
    end_to_end_t: float = 0.0
    # Bytes
    bytes_sent: int = 0
    bytes_recv: int = 0
    bytes_recv_predicted: int = 0
    # Verification
    mse_w1: float = 0.0
    mse_w3: float = 0.0
    mse_w2: float = 0.0
    ok: bool = True
```

- [ ] **Step 4: Implement `Coordinator.run_round` and helpers**

Append to `Coordinator` class (after `close`):

```python
    def reproduce_input_cpu(self, input_seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(input_seed)
        return torch.randn(self.config.batch, self.config.seq,
                            self.config.hidden,
                            dtype=torch.float32, generator=gen)

    def predicted_recv_bytes(self) -> int:
        # 3 ACTIVATION frames (3D tensors) + 1 FORWARD_DONE frame
        frame_hdr = 8
        activation_body_hdr = 11 + 4 * 3  # Q+B+B+B + 3 shape u32
        done_body = 16
        dtype_size = _DTYPE_SIZE[self.config.wire_dtype]
        cfg = self.config
        y1_bytes = cfg.batch * cfg.seq * cfg.inter * dtype_size
        y3_bytes = cfg.batch * cfg.seq * cfg.inter * dtype_size
        y2_bytes = cfg.batch * cfg.seq * cfg.hidden * dtype_size
        return 3 * (frame_hdr + activation_body_hdr) + (frame_hdr + done_body) \
               + y1_bytes + y3_bytes + y2_bytes

    def run_round(self, request_id: int, input_seed: int) -> RoundMetrics:
        assert self.sock is not None, "call connect_and_load() first"
        rm = RoundMetrics(request_id=request_id)
        rm.bytes_recv_predicted = self.predicted_recv_bytes()

        x_cpu = self.reproduce_input_cpu(input_seed)

        # Send FORWARD_REQ
        t_start = time.perf_counter()
        req_body = pack_forward_req(request_id, input_seed,
                                     self.config.batch, self.config.seq)
        rm.bytes_sent = send_msg(self.sock, MSG_FORWARD_REQ, req_body)

        # Receive 3 ACTIVATION + 1 FORWARD_DONE
        t_wire_start = time.perf_counter()
        bytes_recv = 0
        first_act_t: Optional[float] = None
        acts: dict[int, torch.Tensor] = {}
        gpu_forward_t_ms = 0.0
        done = False
        while not done or len(acts) < 3:
            mt, body = recv_msg(self.sock)
            bytes_recv += 8 + len(body)
            if mt == MSG_ACTIVATION:
                if first_act_t is None:
                    first_act_t = time.perf_counter()
                d = unpack_activation(body)
                acts[d["op_tag"]] = d["tensor"]
            elif mt == MSG_FORWARD_DONE:
                d = unpack_forward_done(body)
                gpu_forward_t_ms = d["gpu_forward_t_ms"]
                done = True
            else:
                raise WireProtocolError(f"unexpected msg_type {mt}")
        t_wire_end = time.perf_counter()
        rm.bytes_recv = bytes_recv
        rm.gpu_forward_t = gpu_forward_t_ms
        rm.coord_send_t = ((first_act_t or t_wire_end) - t_start) * 1000.0
        rm.wire_recv_t = (t_wire_end - t_wire_start) * 1000.0

        # Verify (w1 and w3 in parallel; w2 after gated_cpu is built)
        t_verify_start = time.perf_counter()
        y1 = acts[OP_W1].to(torch.float32)
        y3 = acts[OP_W3].to(torch.float32)
        y2 = acts[OP_W2].to(torch.float32)

        def _timed(fn, *args):
            t0 = time.perf_counter()
            mse = fn(*args)
            return mse, (time.perf_counter() - t0) * 1000.0

        f_w1 = self.pool.submit(_timed, slalom_verify,
                                x_cpu, y1, self.s_w1, self.s_tilde_w1)
        f_w3 = self.pool.submit(_timed, slalom_verify,
                                x_cpu, y3, self.s_w3, self.s_tilde_w3)
        rm.mse_w1, rm.cpu_verify_w1_t = f_w1.result()
        rm.mse_w3, rm.cpu_verify_w3_t = f_w3.result()
        gated_cpu = F.silu(y1) * y3
        f_w2 = self.pool.submit(_timed, slalom_verify,
                                gated_cpu, y2, self.s_w2, self.s_tilde_w2)
        rm.mse_w2, rm.cpu_verify_w2_t = f_w2.result()
        rm.cpu_verify_t = (time.perf_counter() - t_verify_start) * 1000.0
        rm.end_to_end_t = (time.perf_counter() - t_start) * 1000.0

        rm.ok = (rm.mse_w1 <= self.threshold
                 and rm.mse_w3 <= self.threshold
                 and rm.mse_w2 <= self.threshold)
        return rm
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k run_round -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): coordinator run_round with parallel SLALOM"
```

---

## Task 11: Multi-round driver + warmup + bytes-predicted accuracy

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_coordinator_run_many_rounds_collects_metrics():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            ms = coord.run_many(rounds=5)
            assert len(ms) == 5
            assert all(r.ok for r in ms)
            assert all(r.bytes_recv > 0 for r in ms)
        finally:
            coord.close()


def test_bytes_recv_matches_predicted_within_one_percent():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            ms = coord.run_many(rounds=3)
            for r in ms:
                rel = abs(r.bytes_recv - r.bytes_recv_predicted) / r.bytes_recv_predicted
                assert rel < 0.01, f"round {r.request_id}: predicted {r.bytes_recv_predicted} got {r.bytes_recv}"
        finally:
            coord.close()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "run_many or bytes_recv_matches" -v`
Expected: FAIL — `run_many` missing.

- [ ] **Step 3: Implement `run_many`**

Append to `Coordinator`:

```python
    def run_many(self, rounds: int, *, input_seed_start: int = 1_000_000) -> list[RoundMetrics]:
        return [self.run_round(i, input_seed_start + i) for i in range(rounds)]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py -k "run_many or bytes_recv_matches" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): multi-round driver with byte-accurate metrics"
```

---

## Task 12: Derived rates + phase breakdown

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_compute_rates_for_round():
    m = _load_module()
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32)
    r = m.RoundMetrics(request_id=0,
                       gpu_forward_t=2.0, wire_recv_t=1.0, cpu_verify_t=4.0,
                       end_to_end_t=8.0, bytes_recv=49000,
                       bytes_recv_predicted=49000)
    rates = m.compute_round_rates(r, cfg, k=m.SLALOM_K)
    assert rates["wire_mbps"] > 0
    assert rates["gpu_gflops"] > 0
    assert rates["verify_gflops"] > 0
    # Phase breakdown
    assert 0 <= rates["gpu_pct"] <= 1
    assert 0 <= rates["wire_pct"] <= 1
    assert 0 <= rates["verify_pct"] <= 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_compute_rates_for_round -v`
Expected: FAIL — `compute_round_rates` not defined.

- [ ] **Step 3: Implement rate computation**

In `examples/multi_machine_ffn.py`, after `RoundMetrics` dataclass, add:

```python
def compute_round_rates(rm: RoundMetrics, cfg: FFNConfig, k: int) -> dict:
    """Derived rates and phase breakdown for one round."""
    B, S, H, I = cfg.batch, cfg.seq, cfg.hidden, cfg.inter
    ffn_flops = 6 * B * S * H * I  # 3 matmuls × 2 × dims per matmul
    slalom_flops = 6 * B * S * (H + I) * k

    def _safe(t_ms: float, flops: float, scale: float = 1e9) -> float:
        return (flops / (t_ms / 1000.0) / scale) if t_ms > 0 else 0.0

    wire_mbps = (rm.bytes_recv / (rm.wire_recv_t / 1000.0) / 1e6) \
                if rm.wire_recv_t > 0 else 0.0
    gpu_gflops = _safe(rm.gpu_forward_t, ffn_flops, 1e9)
    verify_gflops = _safe(rm.cpu_verify_t, slalom_flops, 1e9)

    e2e = rm.end_to_end_t if rm.end_to_end_t > 0 else 1e-9
    return {
        "wire_mbps": wire_mbps,
        "gpu_gflops": gpu_gflops,
        "verify_gflops": verify_gflops,
        "gpu_pct": rm.gpu_forward_t / e2e,
        "wire_pct": rm.wire_recv_t / e2e,
        "verify_pct": rm.cpu_verify_t / e2e,
        "sum_pct": (rm.gpu_forward_t + rm.wire_recv_t + rm.cpu_verify_t) / e2e,
    }
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_compute_rates_for_round -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): derived rates and phase breakdown"
```

---

## Task 13: Summary report formatting

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_format_summary_includes_required_sections():
    m = _load_module()
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32, weight_seed=42)
    rounds_metrics = [
        m.RoundMetrics(request_id=i, gpu_forward_t=2.0, wire_recv_t=1.0,
                       cpu_verify_t=4.0, end_to_end_t=8.0,
                       bytes_recv=49000, bytes_recv_predicted=49000,
                       mse_w1=1e-6, mse_w3=1e-6, mse_w2=1e-6, ok=True)
        for i in range(10)
    ]
    text = m.format_summary(rounds_metrics, cfg, warmup=2, k=m.SLALOM_K)
    for needle in [
        "Multi-Machine FFN Example",
        "Phase timings",
        "Phase breakdown",
        "Rate breakdown",
        "Verification:",
        "Wire bytes",
    ]:
        assert needle in text, f"missing: {needle}"
    assert "rounds passed     8 / 8" in text  # 10 - 2 warmup
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_format_summary_includes_required_sections -v`
Expected: FAIL — `format_summary` missing.

- [ ] **Step 3: Implement `format_summary`**

In `examples/multi_machine_ffn.py`, after `compute_round_rates`, add:

```python
def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(xs, q))


def format_summary(rounds: list[RoundMetrics], cfg: FFNConfig, *,
                    warmup: int, k: int = SLALOM_K) -> str:
    measured = rounds[warmup:] if warmup > 0 else rounds
    if not measured:
        return "(no rounds measured after warmup)"
    e2e = [r.end_to_end_t for r in measured]
    gpu = [r.gpu_forward_t for r in measured]
    wire = [r.wire_recv_t for r in measured]
    verify = [r.cpu_verify_t for r in measured]
    rates = [compute_round_rates(r, cfg, k) for r in measured]
    passed = sum(1 for r in measured if r.ok)
    tokens_per_round = cfg.batch * cfg.seq
    mean_e2e_s = (sum(e2e) / len(e2e)) / 1000.0

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    bytes_recv_mb = _mean([r.bytes_recv for r in measured]) / 1e6
    bytes_pred_mb = _mean([r.bytes_recv_predicted for r in measured]) / 1e6

    return (
        "=== Multi-Machine FFN Example: "
        f"{len(rounds)} rounds (warmup={warmup}) ===\n\n"
        "Config:\n"
        f"  FFN:      SwiGLU  hidden={cfg.hidden}  inter={cfg.inter}  "
        f"dtype={_DTYPE_NAME[cfg.wire_dtype]}  batch×seq={cfg.batch}×{cfg.seq}\n"
        f"  Verify:   SLALOM  k={k}\n\n"
        "End-to-end (ms):\n"
        f"  p50   {_percentile(e2e, 50):.2f}   "
        f"p95   {_percentile(e2e, 95):.2f}   "
        f"mean  {_mean(e2e):.2f}\n"
        f"  Throughput        {1.0/mean_e2e_s:.2f} round/s   "
        f"({tokens_per_round/mean_e2e_s:.1f} tokens/s)\n\n"
        "Phase timings (ms, mean):\n"
        f"  GPU forward       {_mean(gpu):.2f}\n"
        f"  Wire recv         {_mean(wire):.2f}\n"
        f"  CPU SLALOM        {_mean(verify):.2f}     "
        f"(w1={_mean([r.cpu_verify_w1_t for r in measured]):.2f}  "
        f"w3={_mean([r.cpu_verify_w3_t for r in measured]):.2f}  "
        f"w2={_mean([r.cpu_verify_w2_t for r in measured]):.2f})\n\n"
        "Phase breakdown (% of end-to-end, mean):\n"
        f"  gpu_pct           {_mean([r['gpu_pct'] for r in rates])*100:.1f}%\n"
        f"  wire_pct          {_mean([r['wire_pct'] for r in rates])*100:.1f}%\n"
        f"  verify_pct        {_mean([r['verify_pct'] for r in rates])*100:.1f}%\n"
        f"  sum_pct           {_mean([r['sum_pct'] for r in rates])*100:.1f}%\n\n"
        "Rate breakdown:\n"
        f"  Wire throughput   {_mean([r['wire_mbps'] for r in rates]):.1f} MB/s\n"
        f"  GPU compute       {_mean([r['gpu_gflops'] for r in rates])/1000.0:.2f} TFLOPS\n"
        f"  CPU SLALOM        {_mean([r['verify_gflops'] for r in rates]):.2f} GFLOPS\n\n"
        "Wire bytes (per round):\n"
        f"  Predicted         {bytes_pred_mb:.2f} MB\n"
        f"  Measured          {bytes_recv_mb:.2f} MB\n\n"
        "Verification:\n"
        f"  rounds passed     {passed} / {len(measured)}\n"
        f"  mse_w1 p95        {_percentile([r.mse_w1 for r in measured], 95):.2e}\n"
        f"  mse_w3 p95        {_percentile([r.mse_w3 for r in measured], 95):.2e}\n"
        f"  mse_w2 p95        {_percentile([r.mse_w2 for r in measured], 95):.2e}\n"
    )
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_format_summary_includes_required_sections -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): human-readable summary report"
```

---

## Task 14: JSON report writer

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_write_json_report_schema(tmp_path):
    m = _load_module()
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32, weight_seed=42)
    rounds_metrics = [
        m.RoundMetrics(request_id=i, gpu_forward_t=2.0, wire_recv_t=1.0,
                       cpu_verify_t=4.0, end_to_end_t=8.0,
                       bytes_recv=49000, bytes_recv_predicted=49000,
                       mse_w1=1e-6, mse_w3=1e-6, mse_w2=1e-6, ok=True)
        for i in range(3)
    ]
    out = tmp_path / "report.json"
    m.write_json_report(out, rounds_metrics, cfg, warmup=1, k=m.SLALOM_K)
    import json as _json
    data = _json.loads(out.read_text())
    assert "config" in data and "per_round" in data and "summary" in data
    assert len(data["per_round"]) == 3
    assert "mean_round_per_s" in data["summary"]
    assert "p95_end_to_end_ms" in data["summary"]
    assert "mean_wire_mbps" in data["summary"]
    assert "rounds_passed" in data["summary"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_write_json_report_schema -v`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement `write_json_report`**

In `examples/multi_machine_ffn.py`, after `format_summary`, add:

```python
def write_json_report(path, rounds: list[RoundMetrics], cfg: FFNConfig, *,
                      warmup: int, k: int = SLALOM_K) -> None:
    measured = rounds[warmup:] if warmup > 0 else rounds
    rates = [compute_round_rates(r, cfg, k) for r in measured]

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    mean_e2e_ms = _mean([r.end_to_end_t for r in measured]) or 1e-9
    summary = {
        "rounds_total": len(rounds),
        "rounds_warmup": warmup,
        "rounds_measured": len(measured),
        "rounds_passed": sum(1 for r in measured if r.ok),
        "mean_round_per_s": 1000.0 / mean_e2e_ms,
        "p50_end_to_end_ms": _percentile([r.end_to_end_t for r in measured], 50),
        "p95_end_to_end_ms": _percentile([r.end_to_end_t for r in measured], 95),
        "mean_gpu_forward_ms": _mean([r.gpu_forward_t for r in measured]),
        "mean_wire_recv_ms": _mean([r.wire_recv_t for r in measured]),
        "mean_cpu_verify_ms": _mean([r.cpu_verify_t for r in measured]),
        "mean_wire_mbps": _mean([r["wire_mbps"] for r in rates]),
        "mean_gpu_gflops": _mean([r["gpu_gflops"] for r in rates]),
        "mean_verify_gflops": _mean([r["verify_gflops"] for r in rates]),
        "mean_gpu_pct": _mean([r["gpu_pct"] for r in rates]),
        "mean_wire_pct": _mean([r["wire_pct"] for r in rates]),
        "mean_verify_pct": _mean([r["verify_pct"] for r in rates]),
        "mean_sum_pct": _mean([r["sum_pct"] for r in rates]),
    }
    payload = {
        "config": asdict(cfg),
        "per_round": [asdict(r) for r in rounds],
        "summary": summary,
    }
    import pathlib as _p
    _p.Path(path).write_text(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_write_json_report_schema -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): JSON report writer with summary block"
```

---

## Task 15: Loopback launcher

**Files:**
- Modify: `examples/multi_machine_ffn.py`
- Modify: `tests/test_multi_machine_ffn_example.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_loopback_role_runs_end_to_end(tmp_path):
    """`--role loopback` spawns its own worker and runs N rounds clean."""
    json_path = tmp_path / "r.json"
    cmd = [_sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "5", "--warmup", "1",
           "--hidden", "8", "--inter", "16",
           "--batch", "1", "--seq", "4",
           "--wire-dtype", "fp32",
           "--device", "cpu",
           "--json-report", str(json_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "Multi-Machine FFN Example" in res.stdout
    import json as _json
    data = _json.loads(json_path.read_text())
    assert data["summary"]["rounds_passed"] == 4  # 5 - 1 warmup
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_loopback_role_runs_end_to_end -v`
Expected: FAIL — loopback not implemented in main.

- [ ] **Step 3: Implement loopback launcher and finish main()**

Replace the existing `main()` in `examples/multi_machine_ffn.py` with:

```python
# ── Loopback launcher ───────────────────────────────────────────────
def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise TimeoutError(f"worker port {port} did not open within {timeout}s")


def run_coordinator(host: str, port: int, args) -> int:
    cfg = FFNConfig(
        hidden=args.hidden, inter=args.inter,
        batch=args.batch, seq=args.seq,
        wire_dtype=_NAME_TO_DTYPE[args.wire_dtype],
        weight_seed=args.weight_seed,
    )
    coord = Coordinator(host=host, port=port, config=cfg,
                         threshold=args.threshold, k=SLALOM_K)
    try:
        coord.connect_and_load()
        rounds = coord.run_many(rounds=args.rounds)
    finally:
        coord.close()
    print(format_summary(rounds, cfg, warmup=args.warmup, k=SLALOM_K))
    if args.json_report:
        write_json_report(args.json_report, rounds, cfg,
                          warmup=args.warmup, k=SLALOM_K)
    if args.verbose:
        for r in rounds:
            print(f"r={r.request_id} e2e={r.end_to_end_t:.2f} "
                  f"gpu={r.gpu_forward_t:.2f} wire={r.wire_recv_t:.2f} "
                  f"verify={r.cpu_verify_t:.2f} "
                  f"mse={r.mse_w1:.2e}/{r.mse_w3:.2e}/{r.mse_w2:.2e} "
                  f"ok={r.ok}")
    return 0


def launch_loopback(args) -> int:
    port = _pick_free_port()
    worker_cmd = [
        sys.executable, __file__,
        "--role", "worker",
        "--bind", f"127.0.0.1:{port}",
        "--device", args.device,
        "--inject-fault", args.inject_fault,
    ]
    proc = subprocess.Popen(worker_cmd, stderr=subprocess.PIPE)
    try:
        _wait_port(port)
        return run_coordinator("127.0.0.1", port, args)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc.returncode and proc.returncode != 0:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            sys.stderr.write(f"worker exited with {proc.returncode}\n{err}\n")
```

(Leave the existing `main()` for now — Task 16 replaces it with full argparse.)

- [ ] **Step 4: Add temporary loopback dispatch**

Modify the `main()` body to handle loopback:

```python
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["loopback", "worker", "coordinator"],
                   default="loopback")
    p.add_argument("--bind", default="127.0.0.1:9100")
    p.add_argument("--worker-host", default="127.0.0.1")
    p.add_argument("--worker-port", type=int, default=9100)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=11008)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--wire-dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--weight-seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject-fault", default="none",
                   choices=["none", "flip_y1", "scale_y2", "drop_silu"])
    p.add_argument("--json-report", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.role == "worker":
        host, port_s = args.bind.split(":")
        Worker(host, int(port_s), args.device, args.inject_fault).serve_once()
        return 0
    if args.role == "coordinator":
        return run_coordinator(args.worker_host, args.worker_port, args)
    if args.role == "loopback":
        return launch_loopback(args)
    raise ValueError(f"unknown role: {args.role}")
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/test_multi_machine_ffn_example.py::test_loopback_role_runs_end_to_end -v`
Expected: PASS within 30 seconds.

- [ ] **Step 6: Commit**

```bash
git add examples/multi_machine_ffn.py tests/test_multi_machine_ffn_example.py
git commit -m "feat(ffn-example): loopback launcher and full CLI"
```

---

## Task 16: Functional test suite — fill in remaining §11.1 cases

**Files:**
- Modify: `tests/test_multi_machine_ffn_example.py`

This task does not modify the example file — it adds the missing functional tests from spec §11.1 that aren't already covered by Tasks 1-15.

Coverage so far:
- ✅ `test_loopback_passes_clean` → `test_loopback_role_runs_end_to_end` (Task 15)
- ✅ `test_inject_*_caught` → covered by `test_coordinator_run_round_with_flip_y1_caught` (Task 10), plus tag/byte-level tests (Task 8)
- ✅ `test_wire_bytes_predicted_matches` → `test_bytes_recv_matches_predicted_within_one_percent` (Task 11)
- ❌ `test_close_message_shuts_worker` — needs adding
- ❌ `test_wire_dtype_fp32_also_works` — needs adding (we've already been using fp32 in tests, but make it explicit)
- ❌ `test_inject_drop_silu_caught` end-to-end through Coordinator — needs adding
- ❌ `test_inject_scale_y2_caught` end-to-end through Coordinator — needs adding
- ❌ `test_json_report_schema` → covered by Task 14

- [ ] **Step 1: Write the new tests**

Append to `tests/test_multi_machine_ffn_example.py`:

```python
def test_close_message_shuts_worker_within_two_seconds():
    m = _load_module()
    with _WorkerProc() as wp:
        with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
            m.send_msg(sock, m.MSG_LOAD_REQ,
                       m.pack_load_req(hidden=8, inter=16,
                                        weight_seed=42, dtype_id=m.DTYPE_FP32))
            m.recv_msg(sock)
            m.send_msg(sock, m.MSG_CLOSE, b"")
        rc = wp.proc.wait(timeout=2)
        assert rc == 0


def test_wire_dtype_fp16_runs_clean(tmp_path):
    cmd = [_sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "3", "--warmup", "0",
           "--hidden", "16", "--inter", "32",
           "--batch", "1", "--seq", "4",
           "--wire-dtype", "fp16",
           "--device", "cpu",
           "--threshold", "1e-2"]  # fp16 round-trip is noisier
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "rounds passed     3 / 3" in res.stdout


def test_inject_scale_y2_caught_through_coordinator():
    m = _load_module()
    with _WorkerProc(inject_fault="scale_y2") as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            results = coord.run_many(rounds=3)
        finally:
            coord.close()
        assert all(not r.ok for r in results)
        assert all(r.mse_w2 > 1e-3 for r in results)
        # w1, w3 should still pass (only y2 was tampered with)
        assert all(r.mse_w1 < 1e-3 and r.mse_w3 < 1e-3 for r in results)


def test_inject_drop_silu_caught_via_chain():
    m = _load_module()
    with _WorkerProc(inject_fault="drop_silu") as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            results = coord.run_many(rounds=3)
        finally:
            coord.close()
        # Worker's broken SiLU is caught by the third SLALOM check
        # because coord computes SiLU on CPU from its own y1, y3.
        assert all(not r.ok for r in results)
        assert all(r.mse_w2 > 1e-3 for r in results)
        assert all(r.mse_w1 < 1e-3 and r.mse_w3 < 1e-3 for r in results)
```

- [ ] **Step 2: Run full functional suite**

Run: `pytest tests/test_multi_machine_ffn_example.py -v`
Expected: All tests pass; total wall-clock < 60 seconds.

- [ ] **Step 3: Commit**

```bash
git add tests/test_multi_machine_ffn_example.py
git commit -m "test(ffn-example): fill in remaining functional cases"
```

---

## Task 17: Performance test file

**Files:**
- Create: `tests/test_multi_machine_ffn_perf.py`

- [ ] **Step 1: Write the file**

Create `tests/test_multi_machine_ffn_perf.py`:

```python
"""Performance tests for examples/multi_machine_ffn.py.

Gated by the `perf` marker (not run by default CI). Thresholds are
conservative defaults overridable via env vars; tune per hardware.

Run:
    pytest -q tests/test_multi_machine_ffn_perf.py -m perf
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import socket
import subprocess
import sys
import time
from typing import Optional

import pytest
import torch


pytestmark = pytest.mark.perf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_ffn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mmffn", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise TimeoutError(f"worker port {port} did not open within {timeout}s")


class _WorkerProc:
    def __init__(self, device: str, **cli):
        self.port = _pick_free_port()
        self.device = device
        self.cli = cli
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        cmd = [sys.executable, str(EXAMPLE_PATH),
               "--role", "worker",
               "--bind", f"127.0.0.1:{self.port}",
               "--device", self.device]
        for k, v in self.cli.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
        self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        _wait_port(self.port)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# Default shape & thresholds: tuned conservatively for "smallish" GPU/CPU.
# Override via env vars when running on faster hardware.
_HIDDEN = _env_int("FFN_PERF_HIDDEN", 1024)
_INTER  = _env_int("FFN_PERF_INTER",  2752)  # ~2.7x hidden
_BATCH  = _env_int("FFN_PERF_BATCH",  1)
_SEQ    = _env_int("FFN_PERF_SEQ",    128)
_ROUNDS = _env_int("FFN_PERF_ROUNDS", 50)
_WARMUP = _env_int("FFN_PERF_WARMUP", 5)

_MIN_ROUND_PER_S = _env_float("FFN_PERF_MIN_ROUND_PER_S", 5.0)
_MAX_P95_MS      = _env_float("FFN_PERF_MAX_P95_MS", 1000.0)
_MIN_WIRE_MBPS   = _env_float("FFN_PERF_MIN_WIRE_MBPS", 50.0)
_MIN_VERIFY_GFLOPS = _env_float("FFN_PERF_MIN_VERIFY_GFLOPS", 0.5)


def _run_perf(inject_fault: str = "none", device: Optional[str] = None):
    m = _load_module()
    device = device or _device()
    with _WorkerProc(device=device, inject_fault=inject_fault) as wp:
        cfg = m.FFNConfig(hidden=_HIDDEN, inter=_INTER, batch=_BATCH, seq=_SEQ,
                          wire_dtype=m.DTYPE_FP16, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-2)  # fp16 noise floor
        try:
            coord.connect_and_load()
            rounds = coord.run_many(rounds=_ROUNDS)
        finally:
            coord.close()
    measured = rounds[_WARMUP:]
    rates = [m.compute_round_rates(r, cfg, k=m.SLALOM_K) for r in measured]
    return m, cfg, rounds, measured, rates


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _p95(xs):
    import numpy as np
    return float(np.percentile(xs, 95)) if xs else 0.0


def test_steady_state_throughput():
    _m, _cfg, _all, measured, _r = _run_perf()
    mean_e2e = _mean([r.end_to_end_t for r in measured])
    assert mean_e2e > 0
    round_per_s = 1000.0 / mean_e2e
    assert round_per_s >= _MIN_ROUND_PER_S, (
        f"throughput {round_per_s:.2f} round/s < {_MIN_ROUND_PER_S} "
        f"(override with FFN_PERF_MIN_ROUND_PER_S)"
    )


def test_p95_latency_bounded():
    _m, _cfg, _all, measured, _r = _run_perf()
    p95 = _p95([r.end_to_end_t for r in measured])
    assert p95 <= _MAX_P95_MS, (
        f"p95 {p95:.2f}ms > {_MAX_P95_MS}ms "
        f"(override with FFN_PERF_MAX_P95_MS)"
    )


def test_breakdown_sums_close_to_one():
    """Phases are sequential in this example, so sum_pct ≈ 1.0."""
    _m, _cfg, _all, measured, rates = _run_perf()
    mean_sum = _mean([r["sum_pct"] for r in rates])
    assert 0.9 <= mean_sum <= 1.1, (
        f"mean sum_pct {mean_sum:.3f} not in [0.9, 1.1] — "
        f"is the example pipelined unexpectedly?"
    )


def test_wire_throughput_nonzero():
    _m, _cfg, _all, measured, rates = _run_perf()
    mean_mbps = _mean([r["wire_mbps"] for r in rates])
    assert mean_mbps >= _MIN_WIRE_MBPS, (
        f"wire {mean_mbps:.1f} MB/s < {_MIN_WIRE_MBPS} "
        f"(override with FFN_PERF_MIN_WIRE_MBPS)"
    )


def test_gpu_compute_rate_in_valid_range():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _m, _cfg, _all, measured, rates = _run_perf(device="cuda:0")
    mean_gflops = _mean([r["gpu_gflops"] for r in rates])
    assert mean_gflops > 0, "FLOPs computation returned zero"
    # No upper assert (varies wildly by GPU); just lower bound for sanity
    assert mean_gflops > 10.0, f"gpu_gflops {mean_gflops:.1f} suspiciously low"


def test_verify_compute_rate_nonzero():
    _m, _cfg, _all, measured, rates = _run_perf()
    mean_gflops = _mean([r["verify_gflops"] for r in rates])
    assert mean_gflops >= _MIN_VERIFY_GFLOPS, (
        f"verify {mean_gflops:.2f} GFLOPS < {_MIN_VERIFY_GFLOPS} "
        f"(override with FFN_PERF_MIN_VERIFY_GFLOPS)"
    )


def test_warmup_excluded_from_aggregation(tmp_path):
    """Run with high warmup; summary's rounds_measured should reflect it."""
    json_path = tmp_path / "r.json"
    cmd = [sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "10", "--warmup", "4",
           "--hidden", str(_HIDDEN), "--inter", str(_INTER),
           "--batch", str(_BATCH), "--seq", str(_SEQ),
           "--wire-dtype", "fp16", "--device", _device(),
           "--threshold", "1e-2",
           "--json-report", str(json_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stderr={res.stderr}"
    import json
    data = json.loads(json_path.read_text())
    assert data["summary"]["rounds_total"] == 10
    assert data["summary"]["rounds_warmup"] == 4
    assert data["summary"]["rounds_measured"] == 6


def test_clean_run_passes_verification_at_perf_shape():
    """Sanity: the perf-shape run still verifies cleanly (no flakiness)."""
    _m, _cfg, _all, measured, _r = _run_perf()
    failed = [r for r in measured if not r.ok]
    assert not failed, (
        f"{len(failed)}/{len(measured)} rounds failed verification "
        f"at perf shape. First failure mse={failed[0].mse_w1:.2e}/"
        f"{failed[0].mse_w3:.2e}/{failed[0].mse_w2:.2e}"
    )
```

- [ ] **Step 2: Register perf marker in pytest config**

Check whether `tests/conftest.py` or `pyproject.toml` / `pytest.ini` exists; if not, create `tests/conftest.py` with:

```python
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "perf: performance tests (gated, run explicitly)"
    )
```

If `conftest.py` exists, append the `pytest_configure` function (or add the `addinivalue_line` call inside an existing one). Verify with: `grep -r "perf:" tests/ pyproject.toml pytest.ini 2>/dev/null`.

- [ ] **Step 3: Run perf suite once**

Run: `pytest -q tests/test_multi_machine_ffn_perf.py -m perf -v`
Expected: All pass within ~4 minutes (CPU device) or ~1 minute (CUDA).
Note any failures: the most likely cause is a threshold too high for the test machine — adjust env vars and document them in the test docstring.

- [ ] **Step 4: Confirm perf tests skipped without the marker**

Run: `pytest -q tests/test_multi_machine_ffn_perf.py -v`
Expected: "deselected" or 0 tests run (since the marker is module-level).
If pytest reports a warning about an unknown marker, the marker registration in Step 2 was missed — fix and re-run.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_machine_ffn_perf.py tests/conftest.py
git commit -m "test(ffn-example): perf suite (throughput, breakdown, rates)"
```

---

## Task 18: README pointer and final smoke

**Files:**
- Modify: `examples/multi_machine_ffn.py` (just touch the module docstring)
- Verify: existing README is unchanged (this task is verification-only)

- [ ] **Step 1: Verify the example runs from scratch on a clean checkout**

```bash
python examples/multi_machine_ffn.py --rounds 20 --warmup 5 \
    --hidden 256 --inter 704 --batch 1 --seq 32 \
    --wire-dtype fp16 --device cpu --threshold 1e-2
```

Expected output: summary report ending with `rounds passed     15 / 15`.

- [ ] **Step 2: Verify fault injection demos work**

```bash
python examples/multi_machine_ffn.py --rounds 5 --warmup 0 \
    --hidden 256 --inter 704 --batch 1 --seq 32 \
    --wire-dtype fp16 --device cpu --threshold 1e-2 \
    --inject-fault flip_y1
```

Expected: `rounds passed     0 / 5` and `mse_w1 p95` is large (e.g., 1e+00 or above).

Repeat for `--inject-fault scale_y2` (expect `mse_w2` large) and
`--inject-fault drop_silu` (expect `mse_w2` large, `mse_w1`/`mse_w3` small).

- [ ] **Step 3: Verify the full functional suite still passes**

Run: `pytest -q tests/test_multi_machine_ffn_example.py -v`
Expected: All tests pass.

- [ ] **Step 4: Final commit (only if any tweaks were needed)**

If everything ran clean, no commit needed; the plan ends here.
If anything required a small fix during Steps 1-3, commit it:

```bash
git add examples/multi_machine_ffn.py
git commit -m "fix(ffn-example): final smoke fixes"
```

- [ ] **Step 5: Optional — confirm clean working tree**

```bash
git status
```

Expected: `working tree clean` (no uncommitted changes for files under
`examples/multi_machine_ffn.py` or `tests/test_multi_machine_ffn_*.py`).

---

## Self-Review — Spec Coverage

| Spec section | Tasks covering it |
|---|---|
| §1 Goal & Non-Goals | Whole plan (no impl tasks needed) |
| §2 Architecture | Tasks 6, 9 set up two-process model |
| §3 Wire Protocol (frame + bodies) | Tasks 4, 5 |
| §4 Worker (load, forward, fault inject) | Tasks 6, 7, 8 |
| §5 Coordinator (init, connect, run_round) | Tasks 9, 10 |
| §6 SLALOM math | Task 3 |
| §7 Metrics (RoundMetrics) | Task 10 (dataclass), Task 11 (multi-round + predicted bytes) |
| §7.1 Derived rates + breakdown | Task 12 |
| §8 Failure handling | Worker err-on-EOF (Task 6), SLALOM threshold (Task 10) |
| §9 CLI | Task 15 (full argparse in main) |
| §10 Loopback launcher | Task 15 |
| §11.1 Functional tests | Tasks 1-15 (TDD as we go) + Task 16 (fills gaps) |
| §11.2 Perf tests | Task 17 |
| §11.3 Standalone perf invocation | Documented in Task 17 docstring + Task 18 smoke |
| §12 File manifest | All tasks adhere |
| §13 Relationship to MULTI_MACHINE.md | Spec-only |
| §14 Open questions | Surfaced in spec only |

No spec gaps.

## Self-Review — Type/Name Consistency

- `make_weights` returns `(w1, w2, w3)` (Task 2) — matches all later uses
- `FFNConfig.wire_dtype` is an int (DTYPE_FP16=2 etc.), not a string — matches `_TORCH_DTYPE` lookups
- `Coordinator.run_round(request_id, input_seed)` signature matches all callers
- `RoundMetrics` field names (`gpu_forward_t`, `wire_recv_t`, `cpu_verify_t`, etc.) consistent between Task 10 (dataclass), Task 12 (rate compute), Task 13 (summary), Task 14 (JSON)
- `compute_round_rates` returns dict with keys `{wire_mbps, gpu_gflops, verify_gflops, gpu_pct, wire_pct, verify_pct, sum_pct}` — consistent with summary + JSON
- `SLALOM_K = 10` defined Task 1, used everywhere
- `_NAME_TO_DTYPE` defined Task 1, used Task 15 to map CLI string → int

No inconsistencies found.
