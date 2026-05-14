# Multi-Machine Attention & Zimage Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new standalone multi-machine examples (`multi_machine_attn_llama.py`, `multi_machine_attn_zimage.py`, `multi_machine_zimage.py`) plus a shared helper module (`_multi_machine_common.py`), all under `examples/`, mirroring the existing `examples/multi_machine_ffn.py` pattern.

**Architecture:** One shared module exports wire framing, SLALOM helpers, RoPE (Llama + zimage variants), RMSNorm, summary scaffolding, and a loopback launcher. Each example is one CLI binary supporting `--role {loopback,worker,coordinator}`. The existing FFN file is **not modified** — it stays the canonical self-contained reference. Verification is FFN-style: SLALOM the linears, recompute the non-linear core (softmax / silu / RoPE / RMSNorm) on the coordinator from already-verified intermediates.

**Tech Stack:** Python 3.10+, `torch`, `numpy`, stdlib (`socket`, `struct`, `subprocess`, `argparse`, `concurrent.futures`, `dataclasses`, `json`, `time`).

**Spec:** `docs/superpowers/specs/2026-05-14-multi-machine-attn-zimage-design.md` — read this first if any detail below is unclear.

**Execution order:** Phase A (foundation) → Phase B (Llama attn) → Phase C (zimage attn) → Phase D (mini-zimage) → Phase E (parametrized integration tests). User explicitly requested attn examples before mini-zimage.

---

## File Structure

```
examples/_multi_machine_common.py             new  ~350 lines, shared helpers
examples/multi_machine_attn_llama.py          new  ~500 lines, single Llama attn block
examples/multi_machine_attn_zimage.py         new  ~500 lines, single zimage attn block
examples/multi_machine_zimage.py              new  ~700 lines, mini-zimage (N blocks)
tests/test_multi_machine_common.py            new  ~250 lines, unit tests for helpers
tests/test_multi_machine_attn_llama.py        new  ~150 lines, loopback + fault tests
tests/test_multi_machine_attn_zimage.py       new  ~150 lines, loopback + fault tests
tests/test_multi_machine_zimage.py            new  ~200 lines, loopback + fault tests + block propagation
```

`examples/multi_machine_ffn.py` is **not** modified. Existing tests (`tests/test_multi_machine_ffn_example.py`, `tests/test_multi_machine_ffn_perf.py`) are not modified.

The `_` prefix on `_multi_machine_common.py` signals "module-private to this directory"; no public API guarantees, intended for the four `multi_machine_*.py` examples only.

---

## Conventions

- Run tests from repo root: `pytest -q tests/test_multi_machine_common.py` (etc.)
- Each task ends with **one commit**. Commit message format: `feat(mm-common): <subject>`, `feat(mm-attn-llama): <subject>`, `feat(mm-attn-zimage): <subject>`, `feat(mm-zimage): <subject>`, or `test(mm-...): <subject>`.
- All imports go at the top of each file.
- Loopback subprocess tests use `--device cpu` so they run without CUDA.
- Tests load examples dynamically with `importlib.util.spec_from_file_location` (matches `tests/test_multi_machine_ffn_example.py:14-22`).

---

## Phase A — Shared module (`examples/_multi_machine_common.py`)

### Task A1: Wire framing, dtype tables, generic tensor packer

**Files:**
- Create: `examples/_multi_machine_common.py`
- Test: `tests/test_multi_machine_common.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_machine_common.py`:

```python
"""Unit tests for examples/_multi_machine_common.py."""
from __future__ import annotations

import importlib.util
import pathlib
import socket
import struct
import sys
import threading

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "examples" / "_multi_machine_common.py"


def _load_common():
    spec = importlib.util.spec_from_file_location("mmcommon", COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmcommon"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_constants():
    m = _load_common()
    assert m.MSG_LOAD_REQ == 1
    assert m.MSG_LOAD_ACK == 2
    assert m.MSG_FORWARD_REQ == 3
    assert m.MSG_TENSOR == 4
    assert m.MSG_FORWARD_DONE == 5
    assert m.MSG_CLOSE == 6
    assert m.DTYPE_FP32 == 1
    assert m.DTYPE_FP16 == 2
    assert m._DTYPE_SIZE[m.DTYPE_FP32] == 4
    assert m._DTYPE_SIZE[m.DTYPE_FP16] == 2
    assert m._DTYPE_NAME[m.DTYPE_FP16] == "fp16"
    assert m._NAME_TO_DTYPE["fp32"] == m.DTYPE_FP32


def _pair_sockets():
    """Return a connected (client, server) socket pair via loopback."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    accepted: list = []

    def _accept():
        conn, _ = srv.accept()
        accepted.append(conn)

    t = threading.Thread(target=_accept)
    t.start()
    client.connect(("127.0.0.1", port))
    t.join(2.0)
    srv.close()
    return client, accepted[0]


def test_send_recv_msg_round_trip():
    m = _load_common()
    a, b = _pair_sockets()
    try:
        body = b"hello world"
        n = m.send_msg(a, m.MSG_LOAD_ACK, body)
        assert n == 8 + len(body)
        mt, got = m.recv_msg(b)
        assert mt == m.MSG_LOAD_ACK
        assert got == body
    finally:
        a.close(); b.close()


def test_recv_exactly_raises_on_eof():
    m = _load_common()
    a, b = _pair_sockets()
    try:
        a.close()
        with pytest.raises(ConnectionError):
            m.recv_exactly(b, 4)
    finally:
        b.close()


def test_pack_unpack_tensor_round_trip_fp32():
    m = _load_common()
    t = torch.randn(2, 3, 5, dtype=torch.float32)
    body = m.pack_tensor(request_id=42, op_tag=7, tensor=t,
                         wire_dtype_id=m.DTYPE_FP32)
    d = m.unpack_tensor(body)
    assert d["request_id"] == 42
    assert d["op_tag"] == 7
    assert d["dtype_id"] == m.DTYPE_FP32
    assert tuple(d["tensor"].shape) == (2, 3, 5)
    assert torch.equal(d["tensor"], t)


def test_pack_unpack_tensor_round_trip_fp16():
    m = _load_common()
    t = torch.randn(4, 8, dtype=torch.float32)  # source fp32
    body = m.pack_tensor(request_id=9, op_tag=300, tensor=t,
                         wire_dtype_id=m.DTYPE_FP16)
    d = m.unpack_tensor(body)
    assert d["op_tag"] == 300            # u16, must round-trip large values
    assert d["tensor"].dtype == torch.float16
    assert torch.allclose(d["tensor"].float(), t.half().float(), atol=0)


def test_pack_tensor_uses_uint16_op_tag():
    m = _load_common()
    t = torch.zeros(1, dtype=torch.float32)
    body = m.pack_tensor(request_id=0, op_tag=65535, tensor=t,
                         wire_dtype_id=m.DTYPE_FP32)
    # Header layout: <Q H B B  shape...>  → request_id(8) + op_tag(2) + dtype(1) + ndim(1) = 12 bytes
    op_tag = struct.unpack_from("<H", body, 8)[0]
    assert op_tag == 65535
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_multi_machine_common.py -v`
Expected: FAIL — `FileNotFoundError` or `ModuleNotFoundError` (file not yet created).

- [ ] **Step 3: Write `_multi_machine_common.py` skeleton**

Create `examples/_multi_machine_common.py`:

```python
"""Shared helpers for the multi-machine examples.

Used by:
    examples/multi_machine_attn_llama.py
    examples/multi_machine_attn_zimage.py
    examples/multi_machine_zimage.py

The standalone FFN example (examples/multi_machine_ffn.py) does NOT
import from here — it stays self-contained as the canonical reference.
"""
from __future__ import annotations

import socket
import struct
from typing import Optional

import numpy as np
import torch


# ── Wire message types ──────────────────────────────────────────────
MSG_LOAD_REQ      = 1
MSG_LOAD_ACK      = 2
MSG_FORWARD_REQ   = 3
MSG_TENSOR        = 4   # generalized; carries (request_id, op_tag, dtype, shape, payload)
MSG_FORWARD_DONE  = 5
MSG_CLOSE         = 6


# ── Wire dtypes ─────────────────────────────────────────────────────
DTYPE_FP32 = 1
DTYPE_FP16 = 2

_TORCH_DTYPE = {DTYPE_FP32: torch.float32, DTYPE_FP16: torch.float16}
_NUMPY_DTYPE = {DTYPE_FP32: np.float32,    DTYPE_FP16: np.float16}
_DTYPE_SIZE  = {DTYPE_FP32: 4,             DTYPE_FP16: 2}
_DTYPE_NAME  = {DTYPE_FP32: "fp32",        DTYPE_FP16: "fp16"}
_NAME_TO_DTYPE = {v: k for k, v in _DTYPE_NAME.items()}


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


# ── Generic tensor packer ───────────────────────────────────────────
# Header layout: <Q H B B  shape[ndim] (I*ndim)>  + payload
#   request_id (u64) + op_tag (u16) + dtype_id (u8) + ndim (u8)
_TENSOR_HDR_FMT = "<QHBB"
_TENSOR_HDR_SIZE = struct.calcsize(_TENSOR_HDR_FMT)


def pack_tensor(request_id: int, op_tag: int, tensor: torch.Tensor,
                wire_dtype_id: int) -> bytes:
    np_dtype = _NUMPY_DTYPE[wire_dtype_id]
    torch_dtype = _TORCH_DTYPE[wire_dtype_id]
    t = tensor.detach().to(torch_dtype).contiguous().cpu()
    payload = t.numpy().astype(np_dtype, copy=False).tobytes()
    ndim = t.ndim
    shape_bytes = struct.pack(f"<{ndim}I", *t.shape)
    header = struct.pack(_TENSOR_HDR_FMT, request_id, op_tag,
                         wire_dtype_id, ndim)
    return header + shape_bytes + payload


def unpack_tensor(body: bytes) -> dict:
    request_id, op_tag, dtype_id, ndim = struct.unpack_from(
        _TENSOR_HDR_FMT, body, 0)
    off = _TENSOR_HDR_SIZE
    shape = struct.unpack_from(f"<{ndim}I", body, off)
    off += 4 * ndim
    payload = body[off:]
    np_dtype = _NUMPY_DTYPE[dtype_id]
    torch_dtype = _TORCH_DTYPE[dtype_id]
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(arr.copy()).to(torch_dtype)
    return {"request_id": request_id, "op_tag": op_tag,
            "dtype_id": dtype_id, "tensor": tensor}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_common.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/_multi_machine_common.py tests/test_multi_machine_common.py
git commit -m "feat(mm-common): wire framing, dtype tables, generic tensor packer"
```

---

### Task A2: SLALOM helpers

**Files:**
- Modify: `examples/_multi_machine_common.py` (append SLALOM section)
- Modify: `tests/test_multi_machine_common.py` (append SLALOM tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_common.py`:

```python
def test_make_s_deterministic():
    m = _load_common()
    s1 = m.make_s(out_dim=16, k=10, seed=7)
    s2 = m.make_s(out_dim=16, k=10, seed=7)
    s3 = m.make_s(out_dim=16, k=10, seed=8)
    assert torch.equal(s1, s2)
    assert s1.shape == (16, 10)
    assert s1.dtype == torch.float32
    assert not torch.equal(s1, s3)


def test_slalom_verify_passes_for_correct_matmul():
    m = _load_common()
    torch.manual_seed(0)
    W = torch.randn(16, 8)
    x = torch.randn(2, 4, 8)
    y = x @ W.t()
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(W, s)
    assert s_tilde.shape == (8, 10)
    mse = m.slalom_verify(x, y, s, s_tilde)
    assert mse < 1e-10


def test_slalom_verify_safe_returns_inf_on_nan():
    m = _load_common()
    torch.manual_seed(0)
    W = torch.randn(16, 8)
    x = torch.randn(2, 4, 8)
    y = x @ W.t()
    y[0, 0, 0] = float("nan")
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(W, s)
    assert m.slalom_verify_safe(x, y, s, s_tilde) == float("inf")


def test_slalom_verify_fails_for_forged_y():
    m = _load_common()
    torch.manual_seed(0)
    W = torch.randn(16, 8)
    x = torch.randn(2, 4, 8)
    y_forged = torch.randn(2, 4, 16)  # not x @ W.T
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(W, s)
    mse = m.slalom_verify(x, y_forged, s, s_tilde)
    assert mse > 1.0
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_common.py -k slalom -v`
Expected: 4 FAIL — `AttributeError`.

- [ ] **Step 3: Append SLALOM implementation**

Append to `examples/_multi_machine_common.py`:

```python

# ── SLALOM ──────────────────────────────────────────────────────────
SLALOM_K = 10
S_GENERATOR_SEED = 0xDEADBEEF


def make_s(out_dim: int, k: int, seed: int) -> torch.Tensor:
    """Random projection vector. Shape (out_dim, k), fp32 on CPU."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(out_dim, k, dtype=torch.float32, generator=gen)


def precompute_s_tilde(weight: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """For nn.Linear y = x @ weight.T, compute s_tilde = weight.T @ s.

    weight shape: (out, in).  s shape: (out, k).  Returns (in, k) fp32.
    """
    w = weight.detach().to(torch.float32).t().contiguous()
    return w @ s.to(torch.float32)


def slalom_verify(
    x: torch.Tensor, y: torch.Tensor,
    s: torch.Tensor, s_tilde: torch.Tensor,
) -> float:
    """Mean-squared-error between y@s and x@s_tilde."""
    lhs = y.to(torch.float32) @ s
    rhs = x.to(torch.float32) @ s_tilde
    return ((lhs - rhs) ** 2).mean().item()


def slalom_verify_safe(x, y, s, s_tilde) -> float:
    """Like slalom_verify but returns inf if y has any NaN/Inf."""
    if not torch.isfinite(y).all():
        return float("inf")
    return slalom_verify(x, y, s, s_tilde)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_common.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/_multi_machine_common.py tests/test_multi_machine_common.py
git commit -m "feat(mm-common): SLALOM helpers (make_s, precompute_s_tilde, verify_safe)"
```

---

### Task A3: RoPE (Llama and zimage variants) and RMSNorm CPU

**Files:**
- Modify: `examples/_multi_machine_common.py` (append)
- Modify: `tests/test_multi_machine_common.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_common.py`:

```python
def test_rope_llama_round_trip_against_hf():
    """Our apply_rope_llama matches HF apply_rotary_pos_emb."""
    m = _load_common()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    head_dim, max_seq = 64, 32
    cos, sin = m.precompute_rope_cos_sin(head_dim, max_seq, base=500000.0)
    assert cos.shape == (max_seq, head_dim)
    assert sin.shape == (max_seq, head_dim)

    torch.manual_seed(0)
    B, H, S, D = 2, 4, max_seq, head_dim
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    q_ours, k_ours = m.apply_rope_llama(q, k, cos, sin)

    # HF expects (cos, sin) shaped (1, S, D) or (B, S, D)
    cos_hf = cos.unsqueeze(0)
    sin_hf = sin.unsqueeze(0)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)
    assert torch.allclose(q_ours, q_hf, atol=1e-5)
    assert torch.allclose(k_ours, k_hf, atol=1e-5)


def test_rope_zimage_complex_cis_matches_reference():
    """Our apply_rotary_emb_zimage matches the verified_diffusers impl."""
    m = _load_common()
    from verified_diffusers.zimage.attention import apply_rotary_emb

    head_dim, max_seq = 64, 16
    freqs = m.precompute_zimage_freqs_cis(head_dim, max_seq, theta=10000.0)
    assert freqs.shape == (max_seq, head_dim // 2)
    assert freqs.dtype == torch.complex64 or freqs.dtype == torch.complex128

    torch.manual_seed(1)
    B, S, H, D = 2, max_seq, 4, head_dim
    x = torch.randn(B, S, H, D)
    out_ours = m.apply_rotary_emb_zimage(x, freqs)
    # Reference expects freqs shape (S, D/2) and applies it via complex math
    out_ref = apply_rotary_emb(x, freqs)
    assert torch.allclose(out_ours.float(), out_ref.float(), atol=1e-5)


def test_rmsnorm_cpu_matches_torch():
    m = _load_common()
    torch.manual_seed(2)
    x = torch.randn(2, 3, 16)
    weight = torch.randn(16)
    eps = 1e-6

    # No scale_offset → standard RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    out = m.rmsnorm_cpu(x, weight, eps, scale_offset=0.0)
    ref = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * weight
    assert torch.allclose(out, ref, atol=1e-6)

    # scale_offset=1.0 → Qwen3-style (1.0 + weight)
    out2 = m.rmsnorm_cpu(x, weight, eps, scale_offset=1.0)
    ref2 = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)
    assert torch.allclose(out2, ref2, atol=1e-6)

    # weight=None: identity scale
    out3 = m.rmsnorm_cpu(x, None, eps, scale_offset=0.0)
    ref3 = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    assert torch.allclose(out3, ref3, atol=1e-6)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_common.py -k 'rope or rmsnorm' -v`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Append RoPE and RMSNorm implementations**

Append to `examples/_multi_machine_common.py`:

```python

# ── RoPE: Llama / Qwen variant (real cos/sin) ───────────────────────
def precompute_rope_cos_sin(
    head_dim: int, max_seq: int, base: float = 500000.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HuggingFace-compatible RoPE precompute.

    Returns (cos, sin), each of shape (max_seq, head_dim). Each row is
    [cos(t·θ_0), …, cos(t·θ_{D/2-1}), cos(t·θ_0), …, cos(t·θ_{D/2-1})]
    so it can multiply x and rotate_half(x) without broadcasting tricks.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (S, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (S, D)
    return emb.cos().to(device=device, dtype=dtype), \
           emb.sin().to(device=device, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope_llama(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to (B, H, S, D) tensors. cos/sin shape (S, D)."""
    # Add unsqueeze for batch and head dims: (1, 1, S, D)
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_out = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_out, k_out


# ── RoPE: Z-Image variant (complex freqs_cis) ───────────────────────
def precompute_zimage_freqs_cis(
    head_dim: int, max_seq: int, theta: float = 10000.0,
) -> torch.Tensor:
    """Complex freqs_cis for zimage RoPE. Shape (max_seq, head_dim/2)."""
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                          # (S, D/2)
    return torch.polar(torch.ones_like(freqs), freqs)      # complex64


def apply_rotary_emb_zimage(
    x: torch.Tensor, freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply zimage RoPE to (B, S, H, D) tensors.

    Mirrors verified_diffusers/zimage/attention.py:30-43.
    """
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    fc = freqs_cis.unsqueeze(2)                            # (S, 1, D/2)
    x_out = torch.view_as_real(x_c * fc).flatten(3)        # back to (B, S, H, D)
    return x_out.type_as(x)


# ── RMSNorm CPU helper ──────────────────────────────────────────────
def rmsnorm_cpu(
    x: torch.Tensor, weight: Optional[torch.Tensor], eps: float,
    *, scale_offset: float = 0.0,
) -> torch.Tensor:
    """Standard RMSNorm; pass scale_offset=1.0 for Qwen3-style (1+w) scaling."""
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    out = x_f * rms
    if weight is not None:
        out = out * (scale_offset + weight.float())
    return out.to(x.dtype)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_common.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/_multi_machine_common.py tests/test_multi_machine_common.py
git commit -m "feat(mm-common): RoPE (Llama + zimage) and RMSNorm CPU helpers"
```

---

### Task A4: RoundMetricsBase, format_summary, default_slalom_threshold

**Files:**
- Modify: `examples/_multi_machine_common.py` (append)
- Modify: `tests/test_multi_machine_common.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_common.py`:

```python
def test_round_metrics_base_defaults():
    m = _load_common()
    rm = m.RoundMetricsBase(request_id=5)
    assert rm.request_id == 5
    assert rm.gpu_forward_t == 0.0
    assert rm.bytes_recv == 0
    assert rm.recv_tensors == {}
    assert rm.mse == {}
    assert rm.cpu_verify_per_op_t == {}
    assert rm.ok is True


def test_default_slalom_threshold_fp32():
    m = _load_common()
    assert m.default_slalom_threshold(m.DTYPE_FP32, in_dim=1024) == 1e-3
    assert m.default_slalom_threshold(m.DTYPE_FP32, in_dim=4096) == 1e-3


def test_default_slalom_threshold_fp16_scales_with_dim():
    m = _load_common()
    t_small = m.default_slalom_threshold(m.DTYPE_FP16, in_dim=128)
    t_large = m.default_slalom_threshold(m.DTYPE_FP16, in_dim=11008)
    assert t_small == max(1e-3, 128 * 2e-6)
    assert t_large == max(1e-3, 11008 * 2e-6)
    assert t_large > t_small


def test_default_slalom_threshold_custom_slope():
    m = _load_common()
    t = m.default_slalom_threshold(m.DTYPE_FP16, in_dim=4096, fp16_slope=4e-6)
    assert t == max(1e-3, 4096 * 4e-6)


def test_format_summary_basic_smoke():
    m = _load_common()
    rounds = [
        m.RoundMetricsBase(request_id=i, end_to_end_t=10.0, gpu_forward_t=3.0,
                           wire_recv_t=4.0, cpu_verify_t=2.0,
                           bytes_recv=1000, bytes_recv_predicted=1000,
                           mse={1: 1e-7, 2: 1e-7}, ok=True)
        for i in range(5)
    ]
    op_names = {1: "q", 2: "o"}
    s = m.format_summary(rounds, warmup=1, pipelined=False,
                         op_names=op_names,
                         config_lines=["foo: bar"],
                         link_gbps=None)
    assert "End-to-end" in s
    assert "rounds passed" in s
    assert "foo: bar" in s
    assert "q" in s and "o" in s
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_common.py -k 'metrics or threshold or summary' -v`
Expected: FAIL.

- [ ] **Step 3: Append implementations**

Append to `examples/_multi_machine_common.py`:

```python

# ── Metrics ─────────────────────────────────────────────────────────
from dataclasses import dataclass, field


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
    recv_tensors: dict = field(default_factory=dict)
    mse: dict = field(default_factory=dict)
    cpu_verify_per_op_t: dict = field(default_factory=dict)
    ok: bool = True


# ── Threshold ───────────────────────────────────────────────────────
def default_slalom_threshold(
    wire_dtype_id: int, in_dim: int,
    *, floor: float = 1e-3, fp16_slope: float = 2e-6,
) -> float:
    """Pick an MSE threshold above the wire's numeric noise floor."""
    if wire_dtype_id == DTYPE_FP32:
        return 1e-3
    return max(floor, in_dim * fp16_slope)


# ── Summary formatter ───────────────────────────────────────────────
def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(xs, q))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def format_summary(
    rounds: list, *, warmup: int, pipelined: bool,
    op_names: dict[int, str],
    config_lines: list[str] = (),
    link_gbps: Optional[float] = None,
    title: str = "Multi-Machine Example",
) -> str:
    """Render a perf-and-verify summary. Op-tag-agnostic."""
    measured = rounds[warmup:] if warmup > 0 else rounds
    if not measured:
        return "(no rounds measured after warmup)"
    e2e = [r.end_to_end_t for r in measured]
    gpu = [r.gpu_forward_t for r in measured]
    wire = [r.wire_recv_t for r in measured]
    verify = [r.cpu_verify_t for r in measured]
    passed = sum(1 for r in measured if r.ok)
    mean_e2e_s = _mean(e2e) / 1000.0
    bytes_recv_mb = _mean([r.bytes_recv for r in measured]) / 1e6
    bytes_pred_mb = _mean([r.bytes_recv_predicted for r in measured]) / 1e6
    wire_only_ms = _mean(
        [max(0.0, r.wire_recv_t - r.gpu_forward_t) for r in measured])

    # Per-op MSE p95 across the measured rounds
    all_op_tags = sorted({tag for r in measured for tag in r.mse.keys()})
    mse_lines = ""
    for tag in all_op_tags:
        vals = [r.mse[tag] for r in measured if tag in r.mse]
        name = op_names.get(tag, f"op{tag}")
        mse_lines += f"  mse[{name:<12}] p95   {_percentile(vals, 95):.2e}\n"

    cfg_block = "\n".join(f"  {line}" for line in config_lines)

    # Wire estimate
    payload_bits = bytes_recv_mb * 1e6 * 8.0
    eff_gbps = (payload_bits / (wire_only_ms / 1000.0) / 1e9) if wire_only_ms > 0 else 0.0
    ref_gbps = [1.0, 10.0, 25.0]
    if link_gbps and not any(abs(g - link_gbps) < 1e-9 for g in ref_gbps):
        ref_gbps = sorted(ref_gbps + [link_gbps])

    def _ideal_ms(g):
        return payload_bits / (g * 1e9) * 1000.0 if g > 0 else 0.0

    wire_lines = (
        f"  measured            {wire_only_ms:8.2f} ms  "
        f"→  {eff_gbps:6.2f} Gbit/s effective\n"
    )
    for g in ref_gbps:
        tag = "   ← --link-gbps" if link_gbps and abs(g - link_gbps) < 1e-9 else ""
        wire_lines += f"  @ {g:g} Gbit/s ideal  {_ideal_ms(g):8.2f} ms{tag}\n"
    if link_gbps:
        wire_lines += (f"  link efficiency     "
                       f"{eff_gbps / link_gbps * 100:8.1f} %   "
                       f"of nominal {link_gbps:g} Gbit/s\n")

    return (
        f"=== {title}: {len(rounds)} rounds (warmup={warmup}) ===\n\n"
        "Config:\n"
        f"{cfg_block}\n\n"
        "End-to-end (ms):\n"
        f"  p50   {_percentile(e2e, 50):.2f}   "
        f"p95   {_percentile(e2e, 95):.2f}   "
        f"mean  {_mean(e2e):.2f}\n"
        f"  Throughput        {1.0/mean_e2e_s if mean_e2e_s > 0 else 0:.2f} round/s\n\n"
        "Phase timings (ms, mean):\n"
        f"  GPU forward       {_mean(gpu):.2f}\n"
        f"  Wire recv         {_mean(wire):.2f}\n"
        f"  CPU verify        {_mean(verify):.2f}\n\n"
        "Wire bytes (per round):\n"
        f"  Predicted         {bytes_pred_mb:.2f} MB\n"
        f"  Measured          {bytes_recv_mb:.2f} MB\n\n"
        f"Wire estimate (per round, {bytes_recv_mb:.2f} MB payload):\n"
        f"{wire_lines}\n"
        "Verification:\n"
        f"  rounds passed     {passed} / {len(measured)}\n"
        f"{mse_lines}"
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_common.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/_multi_machine_common.py tests/test_multi_machine_common.py
git commit -m "feat(mm-common): RoundMetricsBase, format_summary, default_slalom_threshold"
```

---

### Task A5: Loopback launcher and free-port helper

**Files:**
- Modify: `examples/_multi_machine_common.py` (append)
- Modify: `tests/test_multi_machine_common.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_common.py`:

```python
def test_pick_free_port_returns_usable_port():
    m = _load_common()
    p = m.pick_free_port()
    assert 1024 <= p <= 65535
    # Should be re-bindable (port closed before returning)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", p))
    s.close()


def test_wait_port_succeeds_when_open():
    m = _load_common()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        m.wait_port(port, timeout=2.0)  # should not raise
    finally:
        srv.close()


def test_wait_port_times_out_when_closed():
    m = _load_common()
    # Find a free port and don't bind it
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(TimeoutError):
        m.wait_port(port, timeout=0.5)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_common.py -k 'port' -v`
Expected: FAIL.

- [ ] **Step 3: Append loopback helpers**

Append to `examples/_multi_machine_common.py`:

```python

# ── Loopback launcher helpers ───────────────────────────────────────
import time


def pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise TimeoutError(f"worker port {port} did not open within {timeout}s")


def launch_loopback_worker(
    this_file: str, extra_worker_argv: list[str], *, device: str,
) -> tuple:
    """Spawn `python this_file --role worker --bind 127.0.0.1:<port> ...`

    Returns (subprocess.Popen, port). Caller is responsible for terminating
    the subprocess after the coordinator is done.
    """
    import subprocess
    import sys
    port = pick_free_port()
    cmd = [
        sys.executable, this_file,
        "--role", "worker",
        "--bind", f"127.0.0.1:{port}",
        "--device", device,
        *extra_worker_argv,
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    return proc, port


def cleanup_loopback_worker(proc, *, grace_s: float = 1.0) -> None:
    """Wait briefly for clean exit, then SIGTERM/SIGKILL. Surface unexpected
    non-zero exits (other than -SIGTERM, which we issued ourselves)."""
    import subprocess
    import sys
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    rc = proc.returncode
    if rc is not None and rc != 0 and rc != -15:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        sys.stderr.write(f"worker exited with {rc}\n{err}\n")
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_common.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/_multi_machine_common.py tests/test_multi_machine_common.py
git commit -m "feat(mm-common): loopback launcher helpers (pick_free_port, wait_port, launch/cleanup)"
```

---

## Phase B — Llama attention example (`examples/multi_machine_attn_llama.py`)

### Task B1: Skeleton, config dataclass, weight builder

**Files:**
- Create: `examples/multi_machine_attn_llama.py`
- Create: `tests/test_multi_machine_attn_llama.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_machine_attn_llama.py`:

```python
"""Functional tests for examples/multi_machine_attn_llama.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_attn_llama.py"


def _load():
    spec = importlib.util.spec_from_file_location("mmattnllama", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmattnllama"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_op_tag_constants():
    m = _load()
    assert m.OP_Q == 1
    assert m.OP_K == 2
    assert m.OP_V == 3
    assert m.OP_O == 4


def test_attn_llama_config_defaults():
    m = _load()
    cfg = m.AttnLlamaConfig()
    assert cfg.hidden == 4096
    assert cfg.heads == 32
    assert cfg.kv_heads == 32
    assert cfg.head_dim == 128
    assert cfg.batch == 1
    assert cfg.seq == 512
    assert cfg.rope_base == 500000.0
    assert cfg.weight_seed == 0xC0FFEE


def test_make_attn_weights_shapes_mha():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=64, heads=8, kv_heads=8, head_dim=8,
                            batch=1, seq=4)
    q, k, v, o = m.make_attn_weights(cfg, dtype=torch.float32, device="cpu")
    assert q.weight.shape == (8 * 8, 64)
    assert k.weight.shape == (8 * 8, 64)
    assert v.weight.shape == (8 * 8, 64)
    assert o.weight.shape == (64, 8 * 8)
    assert all(p.bias is None for p in (q, k, v, o))


def test_make_attn_weights_shapes_gqa():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=64, heads=8, kv_heads=2, head_dim=8,
                            batch=1, seq=4)
    q, k, v, o = m.make_attn_weights(cfg, dtype=torch.float32, device="cpu")
    assert q.weight.shape == (8 * 8, 64)        # heads * head_dim
    assert k.weight.shape == (2 * 8, 64)        # kv_heads * head_dim
    assert v.weight.shape == (2 * 8, 64)
    assert o.weight.shape == (64, 8 * 8)


def test_make_attn_weights_deterministic_across_dtype():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=64, heads=8, kv_heads=8, head_dim=8,
                            batch=1, seq=4, weight_seed=42)
    q32, k32, v32, o32 = m.make_attn_weights(cfg, torch.float32, "cpu")
    q16, k16, v16, o16 = m.make_attn_weights(cfg, torch.float16, "cpu")
    assert torch.allclose(q32.weight.half().float(), q16.weight.float())
    assert torch.allclose(o32.weight.half().float(), o16.weight.float())
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write skeleton**

Create `examples/multi_machine_attn_llama.py`:

```python
"""Standalone multi-machine Llama-style attention example.

Coordinator + worker over TCP. Outputs-only wire. SLALOM-verifies the
four linear projections (q, k, v, o); recomputes the non-linear core
(RoPE + softmax + attn matmuls) on the coordinator.

See docs/superpowers/specs/2026-05-14-multi-machine-attn-zimage-design.md.

Run:
    python examples/multi_machine_attn_llama.py                      # loopback
    python examples/multi_machine_attn_llama.py --role worker
    python examples/multi_machine_attn_llama.py --role coordinator \\
        --worker-host 192.168.1.21
"""
from __future__ import annotations

import argparse
import json
import pathlib
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Import shared helpers (sibling file)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _multi_machine_common as mmc                                     # noqa: E402
from _multi_machine_common import (                                     # noqa: E402
    DTYPE_FP16, DTYPE_FP32,
    MSG_CLOSE, MSG_FORWARD_DONE, MSG_FORWARD_REQ,
    MSG_LOAD_ACK, MSG_LOAD_REQ, MSG_TENSOR,
    RoundMetricsBase, SLALOM_K, S_GENERATOR_SEED, WireProtocolError,
    _DTYPE_NAME, _DTYPE_SIZE, _NAME_TO_DTYPE, _TORCH_DTYPE,
    apply_rope_llama, default_slalom_threshold, format_summary,
    launch_loopback_worker, cleanup_loopback_worker, make_s, pack_tensor,
    pick_free_port, precompute_rope_cos_sin, precompute_s_tilde,
    recv_msg, send_msg, slalom_verify, slalom_verify_safe,
    unpack_tensor, wait_port,
)


# ── Op tag namespace ────────────────────────────────────────────────
OP_Q = 1
OP_K = 2
OP_V = 3
OP_O = 4
_OP_NAME = {OP_Q: "q", OP_K: "k", OP_V: "v", OP_O: "o"}


# ── Config ──────────────────────────────────────────────────────────
@dataclass
class AttnLlamaConfig:
    hidden: int = 4096
    heads: int = 32
    kv_heads: int = 32
    head_dim: int = 128
    batch: int = 1
    seq: int = 512
    rope_base: float = 500000.0
    wire_dtype: int = DTYPE_FP16
    weight_seed: int = 0xC0FFEE

    def __post_init__(self):
        assert self.hidden % self.heads == 0, \
            f"hidden ({self.hidden}) must be divisible by heads ({self.heads})"
        assert self.hidden // self.heads == self.head_dim, (
            f"head_dim ({self.head_dim}) must equal hidden//heads "
            f"({self.hidden // self.heads})"
        )
        assert self.heads % self.kv_heads == 0, \
            f"heads ({self.heads}) must be divisible by kv_heads ({self.kv_heads})"

    @property
    def num_kv_groups(self) -> int:
        return self.heads // self.kv_heads


# ── Weight builder ──────────────────────────────────────────────────
def make_attn_weights(
    cfg: AttnLlamaConfig, dtype: torch.dtype, device: str | torch.device,
) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    """Build q/k/v/o linear layers (no bias).

    Same recipe as multi_machine_ffn.make_weights: CPU Generator seeded with
    (weight_seed + offset), std=0.02, then .to(device, dtype). Bit-identical
    coordinator-vs-worker weights up to dtype downcast.
    """
    def _lin(in_dim: int, out_dim: int, offset: int) -> nn.Linear:
        gen = torch.Generator(device="cpu").manual_seed(cfg.weight_seed + offset)
        m = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            m.weight.normal_(0.0, 0.02, generator=gen)
        return m.to(device=device, dtype=dtype)

    q = _lin(cfg.hidden, cfg.heads * cfg.head_dim,    offset=0)
    k = _lin(cfg.hidden, cfg.kv_heads * cfg.head_dim, offset=1)
    v = _lin(cfg.hidden, cfg.kv_heads * cfg.head_dim, offset=2)
    o = _lin(cfg.heads * cfg.head_dim, cfg.hidden,    offset=3)
    return q, k, v, o
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_llama.py tests/test_multi_machine_attn_llama.py
git commit -m "feat(mm-attn-llama): skeleton, AttnLlamaConfig, make_attn_weights"
```

---

### Task B2: Worker class — LOAD + FORWARD compute + send

**Files:**
- Modify: `examples/multi_machine_attn_llama.py` (append)
- Modify: `tests/test_multi_machine_attn_llama.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_attn_llama.py`:

```python
def test_repeat_kv_helper():
    m = _load()
    x = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).float()
    out = m.repeat_kv(x, n_rep=3)
    assert out.shape == (2, 6, 3, 4)
    # Each kv head should be replicated n_rep times consecutively
    assert torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 1], out[:, 2])
    assert torch.equal(out[:, 3], out[:, 4])


def test_worker_forward_matches_reference_compute():
    """Compute the worker's q/k/v/o tensors directly and check against a
    plain torch reference path with RoPE + causal softmax."""
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=32, heads=4, kv_heads=4, head_dim=8,
                            batch=2, seq=6, weight_seed=1)
    q_proj, k_proj, v_proj, o_proj = m.make_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    cos, sin = m.precompute_rope_cos_sin(cfg.head_dim, cfg.seq, cfg.rope_base)

    torch.manual_seed(0)
    x = torch.randn(cfg.batch, cfg.seq, cfg.hidden)
    q_raw, k_raw, v_raw, o_raw = m.compute_attn_forward(
        x, q_proj, k_proj, v_proj, o_proj, cos, sin, cfg)

    # Reference path
    q = q_proj(x).view(cfg.batch, cfg.seq, cfg.heads, cfg.head_dim).transpose(1, 2)
    k = k_proj(x).view(cfg.batch, cfg.seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    v = v_proj(x).view(cfg.batch, cfg.seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    q_rope, k_rope = m.apply_rope_llama(q, k, cos, sin)
    k_rep = m.repeat_kv(k_rope, cfg.num_kv_groups)
    v_rep = m.repeat_kv(v, cfg.num_kv_groups)
    scores = q_rope @ k_rep.transpose(-2, -1) * (cfg.head_dim ** -0.5)
    causal = torch.triu(torch.full((cfg.seq, cfg.seq), float("-inf")), diagonal=1)
    scores = scores + causal
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ v_rep).transpose(1, 2).reshape(cfg.batch, cfg.seq, cfg.hidden)
    o_ref = o_proj(attn_out)

    assert torch.allclose(q_raw, q_proj(x), atol=1e-5)
    assert torch.allclose(k_raw, k_proj(x), atol=1e-5)
    assert torch.allclose(v_raw, v_proj(x), atol=1e-5)
    assert torch.allclose(o_raw, o_ref, atol=1e-4)
```

(Add `import torch.nn.functional as F` at top of test file if not already.)

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -k 'repeat_kv or worker_forward' -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'repeat_kv'`.

- [ ] **Step 3: Append worker compute helpers**

Append to `examples/multi_machine_attn_llama.py`:

```python

# ── repeat_kv (HF compat) ───────────────────────────────────────────
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return expanded.reshape(b, h * n_rep, s, d)


# ── Causal mask ─────────────────────────────────────────────────────
def causal_mask(seq: int, device, dtype) -> torch.Tensor:
    return torch.triu(
        torch.full((seq, seq), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )


# ── Forward compute (shared by Worker.GPU and Coordinator-recompute paths) ──
def compute_attn_forward(
    x: torch.Tensor,
    q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear,
    cos: torch.Tensor, sin: torch.Tensor, cfg: AttnLlamaConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full attention forward and return (q_raw, k_raw, v_raw, output).

    Only the "raw" linear outputs cross the wire — `attn_out` is recomputed
    on the coordinator from the verified q/k/v.
    """
    B, S = cfg.batch, cfg.seq
    q_raw = q_proj(x)
    k_raw = k_proj(x)
    v_raw = v_proj(x)

    q = q_raw.view(B, S, cfg.heads,    cfg.head_dim).transpose(1, 2)
    k = k_raw.view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    v = v_raw.view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)

    q, k = apply_rope_llama(q, k, cos.to(q.dtype), sin.to(q.dtype))
    k = repeat_kv(k, cfg.num_kv_groups)
    v = repeat_kv(v, cfg.num_kv_groups)

    scale = cfg.head_dim ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + causal_mask(S, scores.device, scores.dtype)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ v).transpose(1, 2).contiguous().reshape(B, S, cfg.hidden)
    output = o_proj(attn_out)
    return q_raw, k_raw, v_raw, output
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_llama.py tests/test_multi_machine_attn_llama.py
git commit -m "feat(mm-attn-llama): repeat_kv, causal_mask, compute_attn_forward"
```

---

### Task B3: Worker class (LOAD + FORWARD message handling) and message body packers

**Files:**
- Modify: `examples/multi_machine_attn_llama.py` (append)
- Modify: `tests/test_multi_machine_attn_llama.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_attn_llama.py`:

```python
import socket
import struct
import threading


def _start_worker_thread(worker):
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    return t


def test_worker_load_round_trip():
    """Coordinator-side: send LOAD_REQ, receive LOAD_ACK."""
    m = _load()
    port = m.pick_free_port()
    cfg_kwargs = dict(hidden=32, heads=4, kv_heads=4, head_dim=8,
                      batch=2, seq=6, weight_seed=99)
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port,
                      device="cpu", inject_fault="none",
                      pipeline=False, quiet=True)
    t = _start_worker_thread(worker)
    m.wait_port(port, timeout=3.0)

    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = m.pack_load_req(
        hidden=cfg_kwargs["hidden"], heads=cfg_kwargs["heads"],
        kv_heads=cfg_kwargs["kv_heads"], head_dim=cfg_kwargs["head_dim"],
        rope_base_e9=int(500000.0 * 1e3),  # encoded as int (kHz form)
        weight_seed=cfg_kwargs["weight_seed"],
        dtype_id=m.DTYPE_FP32,
    )
    m.send_msg(sock, m.MSG_LOAD_REQ, body)
    mt, ack = m.recv_msg(sock)
    assert mt == m.MSG_LOAD_ACK
    assert m.unpack_load_ack(ack)["status"] == 0

    m.send_msg(sock, m.MSG_CLOSE, b"")
    sock.close()
    t.join(2.0)
    assert worker.hidden == cfg_kwargs["hidden"]
    assert worker.heads == cfg_kwargs["heads"]
    assert worker.kv_heads == cfg_kwargs["kv_heads"]
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -k 'worker_load_round_trip' -v`
Expected: FAIL — `AttributeError` for `pack_load_req` or `Worker`.

- [ ] **Step 3: Append message body packers and Worker class**

Append to `examples/multi_machine_attn_llama.py`:

```python

# ── Wire message bodies ─────────────────────────────────────────────
# LOAD_REQ body: <I I I I I I B>
#   hidden, heads, kv_heads, head_dim, rope_base_milli (int(base*1000)),
#   weight_seed, dtype_id
_LOAD_REQ_FMT = "<IIIIIIB"


def pack_load_req(hidden: int, heads: int, kv_heads: int, head_dim: int,
                  rope_base_e9: int, weight_seed: int, dtype_id: int) -> bytes:
    return struct.pack(_LOAD_REQ_FMT, hidden, heads, kv_heads, head_dim,
                       rope_base_e9, weight_seed, dtype_id)


def unpack_load_req(body: bytes) -> dict:
    h, hd, kvh, hdim, rope_e9, seed, dtype = struct.unpack(_LOAD_REQ_FMT, body)
    return {"hidden": h, "heads": hd, "kv_heads": kvh, "head_dim": hdim,
            "rope_base": rope_e9 / 1000.0, "weight_seed": seed,
            "dtype_id": dtype}


def pack_load_ack(status: int) -> bytes:
    return struct.pack("<B", status)


def unpack_load_ack(body: bytes) -> dict:
    return {"status": struct.unpack("<B", body)[0]}


# FORWARD_REQ body: <Q I I I>  request_id, input_seed, batch, seq
_FWD_REQ_FMT = "<QIII"


def pack_forward_req(request_id: int, input_seed: int,
                     batch: int, seq: int) -> bytes:
    return struct.pack(_FWD_REQ_FMT, request_id, input_seed, batch, seq)


def unpack_forward_req(body: bytes) -> dict:
    rid, seed, b, s = struct.unpack(_FWD_REQ_FMT, body)
    return {"request_id": rid, "input_seed": seed, "batch": b, "seq": s}


def pack_forward_done(request_id: int, gpu_t_ms: float) -> bytes:
    return struct.pack("<Qd", request_id, gpu_t_ms)


def unpack_forward_done(body: bytes) -> dict:
    rid, t = struct.unpack("<Qd", body)
    return {"request_id": rid, "gpu_forward_t_ms": t}


# ── Worker ──────────────────────────────────────────────────────────
class Worker:
    """Untrusted GPU host. Serves one coordinator at a time."""

    def __init__(self, bind_host: str, bind_port: int, device: str,
                 inject_fault: str = "none", pipeline: bool = False,
                 quiet: bool = False):
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.device = torch.device(device)
        self.inject_fault = inject_fault
        self.pipeline = pipeline
        self.quiet = quiet
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.hidden = self.heads = self.kv_heads = self.head_dim = 0
        self.rope_base = 0.0
        self.cos = self.sin = None
        self.wire_dtype_id = DTYPE_FP16
        self.compute_dtype = torch.float16
        self.cfg: Optional[AttnLlamaConfig] = None
        self._round_count = 0

    def _log(self, msg: str) -> None:
        if self.quiet:
            return
        print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def serve_once(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.bind_port))
        srv.listen(8)
        dev_extra = (f" ({torch.cuda.get_device_name(self.device)})"
                     if self.device.type == "cuda" else "")
        self._log(f"listening on {self.bind_host}:{self.bind_port}  "
                  f"device={self.device}{dev_extra}  fault={self.inject_fault}  "
                  f"pipeline={self.pipeline}")
        try:
            while True:
                sock, addr = srv.accept()
                self._log(f"connection from {addr[0]}:{addr[1]}")
                try:
                    handled = self._serve_session(sock)
                finally:
                    sock.close()
                if handled:
                    self._log(f"session ended (served {self._round_count} round(s)); exiting")
                    return
                self._log("(empty probe connection — still waiting for a coordinator…)")
        finally:
            srv.close()

    def _serve_session(self, sock: socket.socket) -> bool:
        handled = False
        while True:
            try:
                msg_type, body = recv_msg(sock)
            except (ConnectionError, OSError):
                return handled
            handled = True
            if msg_type == MSG_LOAD_REQ:
                self._handle_load(sock, unpack_load_req(body))
            elif msg_type == MSG_FORWARD_REQ:
                self._handle_forward(sock, unpack_forward_req(body))
            elif msg_type == MSG_CLOSE:
                self._log("coordinator sent CLOSE")
                return handled
            else:
                raise WireProtocolError(
                    f"unexpected msg_type {msg_type} on attn-llama worker")

    def _handle_load(self, sock, fields: dict) -> None:
        self.hidden = fields["hidden"]
        self.heads = fields["heads"]
        self.kv_heads = fields["kv_heads"]
        self.head_dim = fields["head_dim"]
        self.rope_base = fields["rope_base"]
        self.wire_dtype_id = fields["dtype_id"]
        self.compute_dtype = _TORCH_DTYPE[self.wire_dtype_id]
        self.cfg = AttnLlamaConfig(
            hidden=self.hidden, heads=self.heads, kv_heads=self.kv_heads,
            head_dim=self.head_dim, batch=1, seq=1,  # placeholders; per-FORWARD overrides
            rope_base=self.rope_base, wire_dtype=self.wire_dtype_id,
            weight_seed=fields["weight_seed"],
        )
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = make_attn_weights(
            self.cfg, dtype=self.compute_dtype, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        send_msg(sock, MSG_LOAD_ACK, pack_load_ack(0))
        self._log(f"LOAD: hidden={self.hidden} heads={self.heads}/"
                  f"kv={self.kv_heads} head_dim={self.head_dim} "
                  f"wire={_DTYPE_NAME[self.wire_dtype_id]}")

    def _handle_forward(self, sock, fields: dict) -> None:
        request_id = fields["request_id"]
        input_seed = fields["input_seed"]
        B, S = fields["batch"], fields["seq"]
        # Build a per-FORWARD config view (heads/kv/head_dim never change after LOAD)
        cfg = AttnLlamaConfig(
            hidden=self.hidden, heads=self.heads, kv_heads=self.kv_heads,
            head_dim=self.head_dim, batch=B, seq=S, rope_base=self.rope_base,
            wire_dtype=self.wire_dtype_id, weight_seed=self.cfg.weight_seed)
        cos, sin = precompute_rope_cos_sin(
            self.head_dim, S, base=self.rope_base,
            device=self.device, dtype=self.compute_dtype)

        gen = torch.Generator(device="cpu").manual_seed(input_seed)
        x_cpu = torch.randn(B, S, self.hidden, dtype=torch.float32, generator=gen)
        x = x_cpu.to(device=self.device, dtype=self.compute_dtype)

        t0 = time.perf_counter()
        q_raw, k_raw, v_raw, output = compute_attn_forward(
            x, self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            cos, sin, cfg)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_t_ms = (time.perf_counter() - t0) * 1000.0

        q_raw, k_raw, v_raw, output = self._apply_fault(
            q_raw, k_raw, v_raw, output, x, cos, sin, cfg)

        if self.pipeline:
            self._send_pipelined(sock, request_id,
                                 q_raw, k_raw, v_raw, output, gpu_t_ms)
        else:
            for tag, t in ((OP_Q, q_raw), (OP_K, k_raw),
                           (OP_V, v_raw), (OP_O, output)):
                send_msg(sock, MSG_TENSOR,
                         pack_tensor(request_id, tag, t, self.wire_dtype_id))
            send_msg(sock, MSG_FORWARD_DONE,
                     pack_forward_done(request_id, gpu_t_ms))

        self._round_count += 1
        sent_mb = sum(t.numel() for t in (q_raw, k_raw, v_raw, output)) \
                  * _DTYPE_SIZE[self.wire_dtype_id] / 1e6
        fault = f"  [INJECTED FAULT: {self.inject_fault}]" if self.inject_fault != "none" else ""
        mode = "pipelined" if self.pipeline else "sequential"
        self._log(f"round #{self._round_count} (req {request_id}): "
                  f"x[{B},{S},{self.hidden}] → forward {gpu_t_ms:.2f} ms on "
                  f"{self.device.type}, sent {sent_mb:.2f} MB ({mode}){fault}")

    def _send_pipelined(self, sock, request_id, q, k, v, o, gpu_t_ms) -> None:
        send_q: queue.Queue = queue.Queue()
        sender_exc: list = []

        def _sender():
            try:
                while True:
                    item = send_q.get()
                    if item is None:
                        return
                    mtype, body = item
                    send_msg(sock, mtype, body)
            except Exception as e:
                sender_exc.append(e)

        t = threading.Thread(target=_sender, daemon=True)
        t.start()
        for tag, ten in ((OP_Q, q), (OP_K, k), (OP_V, v), (OP_O, o)):
            send_q.put((MSG_TENSOR,
                        pack_tensor(request_id, tag, ten, self.wire_dtype_id)))
        send_q.put((MSG_FORWARD_DONE, pack_forward_done(request_id, gpu_t_ms)))
        send_q.put(None)
        t.join()
        if sender_exc:
            raise sender_exc[0]

    def _apply_fault(self, q, k, v, o, x, cos, sin, cfg):
        if self.inject_fault == "none":
            return q, k, v, o
        if self.inject_fault == "flip_v":
            return q, k, -v, o
        if self.inject_fault == "scale_o":
            return q, k, v, o * 1.01
        if self.inject_fault == "drop_softmax":
            # Recompute o without softmax: probs := scores (no normalization)
            B, S = cfg.batch, cfg.seq
            qv = q.view(B, S, cfg.heads, cfg.head_dim).transpose(1, 2)
            kv = k.view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
            vv = v.view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
            qv, kv = apply_rope_llama(qv, kv, cos.to(qv.dtype), sin.to(qv.dtype))
            kv = repeat_kv(kv, cfg.num_kv_groups)
            vv = repeat_kv(vv, cfg.num_kv_groups)
            scores = qv @ kv.transpose(-2, -1) * (cfg.head_dim ** -0.5)
            attn_out = (scores @ vv).transpose(1, 2).reshape(B, S, cfg.hidden)
            return q, k, v, self.o_proj(attn_out)
        if self.inject_fault == "drop_rope":
            B, S = cfg.batch, cfg.seq
            qv = q.view(B, S, cfg.heads, cfg.head_dim).transpose(1, 2)
            kv = k.view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
            vv = v.view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
            # NO RoPE applied
            kv = repeat_kv(kv, cfg.num_kv_groups)
            vv = repeat_kv(vv, cfg.num_kv_groups)
            scores = qv @ kv.transpose(-2, -1) * (cfg.head_dim ** -0.5)
            scores = scores + causal_mask(S, scores.device, scores.dtype)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            attn_out = (probs @ vv).transpose(1, 2).reshape(B, S, cfg.hidden)
            return q, k, v, self.o_proj(attn_out)
        raise ValueError(f"unknown inject_fault: {self.inject_fault}")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_llama.py tests/test_multi_machine_attn_llama.py
git commit -m "feat(mm-attn-llama): Worker class with LOAD/FORWARD/CLOSE handling and fault injection"
```

---

### Task B4: Coordinator class — connect, verify, run_round

**Files:**
- Modify: `examples/multi_machine_attn_llama.py` (append)
- Modify: `tests/test_multi_machine_attn_llama.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_attn_llama.py`:

```python
def test_coordinator_loopback_round_passes_with_no_fault():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=32, heads=4, kv_heads=4, head_dim=8,
                            batch=2, seq=8, wire_dtype=m.DTYPE_FP32,
                            weight_seed=7)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", pipeline=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K, pipeline=False)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is True
    assert rm.mse[m.OP_Q] < 1e-3
    assert rm.mse[m.OP_O] < 1e-3
    assert rm.bytes_recv == rm.bytes_recv_predicted


def test_coordinator_loopback_detects_flip_v_fault():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=32, heads=4, kv_heads=4, head_dim=8,
                            batch=2, seq=8, wire_dtype=m.DTYPE_FP32,
                            weight_seed=7)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="flip_v", pipeline=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K, pipeline=False)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is False
    assert rm.mse[m.OP_V] > 1e-2  # well above the 1e-3 threshold
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -k coordinator -v`
Expected: FAIL — `AttributeError: Coordinator`.

- [ ] **Step 3: Append Coordinator class**

Append to `examples/multi_machine_attn_llama.py`:

```python

# ── Coordinator ─────────────────────────────────────────────────────
class Coordinator:
    """Trusted host. Owns SLALOM state and verifies every linear output."""

    def __init__(self, host: str, port: int, config: AttnLlamaConfig,
                 threshold: float, k: int = SLALOM_K, pipeline: bool = False,
                 o_threshold: Optional[float] = None):
        self.host = host
        self.port = port
        self.config = config
        self.threshold = threshold
        self.o_threshold = o_threshold if o_threshold is not None else threshold
        self.k = k
        self.pipeline = pipeline
        # CPU fp32 weights
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = make_attn_weights(
            config, dtype=torch.float32, device="cpu")
        # SLALOM keys per op
        cfg_in_qkv = config.hidden
        cfg_out_q  = config.heads * config.head_dim
        cfg_out_kv = config.kv_heads * config.head_dim
        cfg_out_o  = config.hidden
        self.s_q  = make_s(cfg_out_q,  k, seed=S_GENERATOR_SEED + 1)
        self.s_k  = make_s(cfg_out_kv, k, seed=S_GENERATOR_SEED + 2)
        self.s_v  = make_s(cfg_out_kv, k, seed=S_GENERATOR_SEED + 3)
        self.s_o  = make_s(cfg_out_o,  k, seed=S_GENERATOR_SEED + 4)
        self.s_tilde_q = precompute_s_tilde(self.q_proj.weight, self.s_q)
        self.s_tilde_k = precompute_s_tilde(self.k_proj.weight, self.s_k)
        self.s_tilde_v = precompute_s_tilde(self.v_proj.weight, self.s_v)
        self.s_tilde_o = precompute_s_tilde(self.o_proj.weight, self.s_o)
        # Per-(seq) RoPE precompute is done lazily
        self._rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.sock: Optional[socket.socket] = None
        self.pool = ThreadPoolExecutor(max_workers=4)

    def _get_rope_cpu(self, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq not in self._rope_cache:
            self._rope_cache[seq] = precompute_rope_cos_sin(
                self.config.head_dim, seq, base=self.config.rope_base)
        return self._rope_cache[seq]

    def connect_and_load(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        body = pack_load_req(
            hidden=self.config.hidden, heads=self.config.heads,
            kv_heads=self.config.kv_heads, head_dim=self.config.head_dim,
            rope_base_e9=int(self.config.rope_base * 1000),
            weight_seed=self.config.weight_seed,
            dtype_id=self.config.wire_dtype,
        )
        send_msg(self.sock, MSG_LOAD_REQ, body)
        mt, ack = recv_msg(self.sock)
        if mt != MSG_LOAD_ACK:
            raise WireProtocolError(f"expected LOAD_ACK, got {mt}")
        if unpack_load_ack(ack)["status"] != 0:
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

    def reproduce_input_cpu(self, input_seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(input_seed)
        return torch.randn(self.config.batch, self.config.seq,
                           self.config.hidden, dtype=torch.float32, generator=gen)

    def predicted_recv_bytes(self) -> int:
        # 4 TENSOR + 1 FORWARD_DONE
        frame_hdr = 8
        # tensor body header = struct.calcsize("<QHBB") + ndim*4 (3 dims)
        tensor_body_hdr = struct.calcsize("<QHBB") + 4 * 3
        done_body = struct.calcsize("<Qd")
        ds = _DTYPE_SIZE[self.config.wire_dtype]
        cfg = self.config
        bytes_q = cfg.batch * cfg.seq * cfg.heads * cfg.head_dim * ds
        bytes_k = cfg.batch * cfg.seq * cfg.kv_heads * cfg.head_dim * ds
        bytes_v = bytes_k
        bytes_o = cfg.batch * cfg.seq * cfg.hidden * ds
        return (4 * (frame_hdr + tensor_body_hdr) + (frame_hdr + done_body)
                + bytes_q + bytes_k + bytes_v + bytes_o)

    def _expected_shape(self, op_tag: int) -> tuple:
        cfg = self.config
        if op_tag == OP_Q:
            return (cfg.batch, cfg.seq, cfg.heads * cfg.head_dim)
        if op_tag in (OP_K, OP_V):
            return (cfg.batch, cfg.seq, cfg.kv_heads * cfg.head_dim)
        if op_tag == OP_O:
            return (cfg.batch, cfg.seq, cfg.hidden)
        raise WireProtocolError(f"unknown op_tag {op_tag}")

    def _recompute_attn_cpu(
        self, q_cpu: torch.Tensor, k_cpu: torch.Tensor, v_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Run attention math on CPU using fp32 q/k/v from worker."""
        cfg = self.config
        cos, sin = self._get_rope_cpu(cfg.seq)
        q = q_cpu.view(cfg.batch, cfg.seq, cfg.heads,    cfg.head_dim).transpose(1, 2)
        k = k_cpu.view(cfg.batch, cfg.seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = v_cpu.view(cfg.batch, cfg.seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = apply_rope_llama(q, k, cos, sin)
        k = repeat_kv(k, cfg.num_kv_groups)
        v = repeat_kv(v, cfg.num_kv_groups)
        scores = q @ k.transpose(-2, -1) * (cfg.head_dim ** -0.5)
        scores = scores + causal_mask(cfg.seq, scores.device, scores.dtype)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_out = (probs @ v).transpose(1, 2).contiguous().reshape(
            cfg.batch, cfg.seq, cfg.hidden)
        return attn_out

    @staticmethod
    def _timed(fn, *args):
        t0 = time.perf_counter()
        mse = fn(*args)
        return mse, (time.perf_counter() - t0) * 1000.0

    def run_round(self, request_id: int, input_seed: int) -> RoundMetricsBase:
        assert self.sock is not None, "call connect_and_load() first"
        rm = RoundMetricsBase(request_id=request_id)
        rm.bytes_recv_predicted = self.predicted_recv_bytes()

        x_cpu = self.reproduce_input_cpu(input_seed)

        t_start = time.perf_counter()
        rm.bytes_sent = send_msg(
            self.sock, MSG_FORWARD_REQ,
            pack_forward_req(request_id, input_seed,
                             self.config.batch, self.config.seq))

        t_wire_start = time.perf_counter()
        bytes_recv = 0
        acts: dict[int, torch.Tensor] = {}
        gpu_t = 0.0
        done = False
        while not done or len(acts) < 4:
            mt, body = recv_msg(self.sock)
            bytes_recv += 8 + len(body)
            if mt == MSG_TENSOR:
                d = unpack_tensor(body)
                op = d["op_tag"]
                t = d["tensor"]
                if tuple(t.shape) != self._expected_shape(op):
                    raise WireProtocolError(
                        f"op_tag={op}: expected {self._expected_shape(op)}, "
                        f"got {tuple(t.shape)}")
                rm.recv_tensors[op] = {
                    "shape": list(t.shape),
                    "dtype": _DTYPE_NAME[d["dtype_id"]],
                    "bytes": 8 + len(body),
                }
                acts[op] = t.to(torch.float32)
            elif mt == MSG_FORWARD_DONE:
                gpu_t = unpack_forward_done(body)["gpu_forward_t_ms"]
                done = True
            else:
                raise WireProtocolError(f"unexpected msg_type {mt}")
        t_wire_end = time.perf_counter()
        rm.bytes_recv = bytes_recv
        rm.gpu_forward_t = gpu_t
        rm.wire_recv_t = (t_wire_end - t_wire_start) * 1000.0

        # SLALOM-verify q/k/v in parallel; recompute attn on CPU; SLALOM-verify o
        t_v_start = time.perf_counter()
        f_q = self.pool.submit(self._timed, slalom_verify_safe,
                               x_cpu, acts[OP_Q], self.s_q, self.s_tilde_q)
        f_k = self.pool.submit(self._timed, slalom_verify_safe,
                               x_cpu, acts[OP_K], self.s_k, self.s_tilde_k)
        f_v = self.pool.submit(self._timed, slalom_verify_safe,
                               x_cpu, acts[OP_V], self.s_v, self.s_tilde_v)
        rm.mse[OP_Q], rm.cpu_verify_per_op_t[OP_Q] = f_q.result()
        rm.mse[OP_K], rm.cpu_verify_per_op_t[OP_K] = f_k.result()
        rm.mse[OP_V], rm.cpu_verify_per_op_t[OP_V] = f_v.result()
        attn_out_cpu = self._recompute_attn_cpu(acts[OP_Q], acts[OP_K], acts[OP_V])
        rm.mse[OP_O], rm.cpu_verify_per_op_t[OP_O] = self._timed(
            slalom_verify_safe, attn_out_cpu, acts[OP_O], self.s_o, self.s_tilde_o)
        rm.cpu_verify_t = (time.perf_counter() - t_v_start) * 1000.0

        rm.end_to_end_t = (time.perf_counter() - t_start) * 1000.0
        rm.ok = (
            rm.mse[OP_Q] <= self.threshold and
            rm.mse[OP_K] <= self.threshold and
            rm.mse[OP_V] <= self.threshold and
            rm.mse[OP_O] <= self.o_threshold
        )
        return rm

    def run_many(self, rounds: int, *, input_seed_start: int = 1_000_000) -> list:
        return [self.run_round(i, input_seed_start + i) for i in range(rounds)]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_llama.py tests/test_multi_machine_attn_llama.py
git commit -m "feat(mm-attn-llama): Coordinator with SLALOM verify and CPU attention recompute"
```

---

### Task B5: CLI, main(), loopback launcher integration

**Files:**
- Modify: `examples/multi_machine_attn_llama.py` (append)
- Modify: `tests/test_multi_machine_attn_llama.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_attn_llama.py`:

```python
import subprocess


def test_loopback_subprocess_smoke():
    """Run the example end-to-end as a subprocess via --role loopback."""
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--hidden", "32", "--heads", "4", "--kv-heads", "4",
         "--head-dim", "8", "--batch", "2", "--seq", "8",
         "--wire-dtype", "fp32", "--rounds", "5", "--warmup", "1"],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}\nstdout:\n{out.stdout}"
    assert "rounds passed     4 / 4" in out.stdout
    assert "End-to-end" in out.stdout


def test_loopback_subprocess_fault_detection():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--hidden", "32", "--heads", "4", "--kv-heads", "4",
         "--head-dim", "8", "--batch", "2", "--seq", "8",
         "--wire-dtype", "fp32", "--rounds", "3", "--warmup", "0",
         "--inject-fault", "scale_o"],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0
    # All measured rounds should fail
    assert "rounds passed     0 / 3" in out.stdout
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -k subprocess -v`
Expected: FAIL — example has no `main()` yet.

- [ ] **Step 3: Append CLI / main / loopback launcher**

Append to `examples/multi_machine_attn_llama.py`:

```python

# ── CLI driver ──────────────────────────────────────────────────────
def run_coordinator(host: str, port: int, args) -> int:
    cfg = AttnLlamaConfig(
        hidden=args.hidden, heads=args.heads, kv_heads=args.kv_heads,
        head_dim=args.head_dim, batch=args.batch, seq=args.seq,
        rope_base=args.rope_base,
        wire_dtype=_NAME_TO_DTYPE[args.wire_dtype],
        weight_seed=args.weight_seed,
    )
    threshold = args.threshold
    if threshold is None:
        threshold = default_slalom_threshold(cfg.wire_dtype, cfg.hidden)
    o_threshold = default_slalom_threshold(
        cfg.wire_dtype, cfg.hidden, fp16_slope=4e-6)
    coord = Coordinator(host=host, port=port, config=cfg,
                        threshold=threshold, k=SLALOM_K,
                        pipeline=args.pipeline, o_threshold=o_threshold)
    try:
        coord.connect_and_load()
        rounds = coord.run_many(rounds=args.rounds)
    finally:
        coord.close()
    config_lines = [
        f"Attn:     Llama  hidden={cfg.hidden}  heads={cfg.heads}/"
        f"kv={cfg.kv_heads}  head_dim={cfg.head_dim}",
        f"Shape:    batch={cfg.batch}  seq={cfg.seq}  "
        f"dtype={_DTYPE_NAME[cfg.wire_dtype]}",
        f"RoPE:     base={cfg.rope_base}",
        f"Verify:   SLALOM  k={SLALOM_K}  thr_qkv={threshold:.1e}  "
        f"thr_o={o_threshold:.1e}",
        f"Pipeline: {'on' if args.pipeline else 'off'}",
    ]
    print(format_summary(
        rounds, warmup=args.warmup, pipelined=args.pipeline,
        op_names=_OP_NAME, config_lines=config_lines,
        link_gbps=args.link_gbps,
        title="Multi-Machine Llama Attention Example"))
    if args.json_report:
        pathlib.Path(args.json_report).write_text(json.dumps({
            "config": asdict(cfg),
            "per_round": [asdict(r) for r in rounds],
        }, indent=2, default=lambda o: list(o) if isinstance(o, dict) else str(o)))
    return 0


def launch_loopback(args) -> int:
    proc, port = launch_loopback_worker(
        __file__, extra_worker_argv=["--inject-fault", args.inject_fault]
                  + (["--pipeline"] if args.pipeline else []),
        device=args.device,
    )
    try:
        wait_port(port, timeout=10.0)
        return run_coordinator("127.0.0.1", port, args)
    finally:
        cleanup_loopback_worker(proc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["loopback", "worker", "coordinator"],
                   default="loopback")
    p.add_argument("--bind", default="127.0.0.1:9101")
    p.add_argument("--worker-host", default="127.0.0.1")
    p.add_argument("--worker-port", type=int, default=9101)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--rope-base", type=float, default=500000.0)
    p.add_argument("--wire-dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--weight-seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject-fault", default="none",
                   choices=["none", "flip_v", "scale_o", "drop_softmax", "drop_rope"])
    p.add_argument("--json-report", default=None)
    p.add_argument("--link-gbps", type=float, default=None)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.role == "worker":
        host, port_s = args.bind.split(":")
        Worker(host, int(port_s), args.device, args.inject_fault,
               pipeline=args.pipeline, quiet=args.quiet).serve_once()
        return 0
    if args.role == "coordinator":
        return run_coordinator(args.worker_host, args.worker_port, args)
    if args.role == "loopback":
        return launch_loopback(args)
    raise ValueError(f"unknown role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all tests for the example**

Run: `pytest -q tests/test_multi_machine_attn_llama.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_llama.py tests/test_multi_machine_attn_llama.py
git commit -m "feat(mm-attn-llama): CLI, run_coordinator, launch_loopback, main entrypoint"
```

---

## Phase C — Z-Image attention example (`examples/multi_machine_attn_zimage.py`)

### Task C1: Skeleton, config, weights (with QK RMSNorm), forward compute

**Files:**
- Create: `examples/multi_machine_attn_zimage.py`
- Create: `tests/test_multi_machine_attn_zimage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_machine_attn_zimage.py`:

```python
"""Functional tests for examples/multi_machine_attn_zimage.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import threading

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_attn_zimage.py"


def _load():
    spec = importlib.util.spec_from_file_location("mmattnzi", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmattnzi"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_op_tag_constants():
    m = _load()
    assert m.OP_Q == 1 and m.OP_K == 2 and m.OP_V == 3 and m.OP_O == 4


def test_attn_zimage_config_defaults():
    m = _load()
    cfg = m.AttnZimageConfig()
    assert cfg.dim == 1536
    assert cfg.heads == 12
    assert cfg.head_dim == 128
    assert cfg.qk_norm == "rms"


def test_make_zimage_weights_shapes():
    m = _load()
    cfg = m.AttnZimageConfig(dim=64, heads=4, head_dim=16, batch=2, seq=8)
    q, k, v, o, nq, nk = m.make_zimage_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    assert q.weight.shape == (4 * 16, 64)
    assert k.weight.shape == (4 * 16, 64)
    assert v.weight.shape == (4 * 16, 64)
    assert o.weight.shape == (64, 4 * 16)
    assert nq.shape == (16,)   # per head_dim RMSNorm scale
    assert nk.shape == (16,)


def test_make_zimage_weights_norm_skipped_when_none():
    m = _load()
    cfg = m.AttnZimageConfig(dim=64, heads=4, head_dim=16, batch=2, seq=8,
                             qk_norm="none")
    q, k, v, o, nq, nk = m.make_zimage_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    assert nq is None and nk is None


def test_compute_zimage_attn_forward_matches_reference():
    m = _load()
    cfg = m.AttnZimageConfig(dim=32, heads=4, head_dim=8, batch=2, seq=6,
                             qk_norm="rms", weight_seed=3)
    q_proj, k_proj, v_proj, o_proj, nq, nk = m.make_zimage_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    freqs = m.precompute_zimage_freqs_cis(cfg.head_dim, cfg.seq, theta=cfg.rope_theta)

    torch.manual_seed(0)
    x = torch.randn(cfg.batch, cfg.seq, cfg.dim)
    q_raw, k_raw, v_raw, o_raw = m.compute_zimage_attn_forward(
        x, q_proj, k_proj, v_proj, o_proj, nq, nk, freqs, cfg)

    # Reference
    q = q_proj(x).unflatten(-1, (cfg.heads, cfg.head_dim))
    k = k_proj(x).unflatten(-1, (cfg.heads, cfg.head_dim))
    v = v_proj(x).unflatten(-1, (cfg.heads, cfg.head_dim))
    eps = m.ZIMAGE_QK_NORM_EPS
    q = m.rmsnorm_cpu(q, nq, eps, scale_offset=0.0)
    k = m.rmsnorm_cpu(k, nk, eps, scale_offset=0.0)
    q = m.apply_rotary_emb_zimage(q, freqs)
    k = m.apply_rotary_emb_zimage(k, freqs)
    q_t = q.permute(0, 2, 1, 3); k_t = k.permute(0, 2, 1, 3); v_t = v.permute(0, 2, 1, 3)
    scores = q_t @ k_t.transpose(2, 3) * (cfg.head_dim ** -0.5)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ v_t).permute(0, 2, 1, 3).flatten(2, 3)
    o_ref = o_proj(attn_out)

    assert torch.allclose(q_raw, q_proj(x), atol=1e-5)
    assert torch.allclose(o_raw, o_ref, atol=1e-4)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_attn_zimage.py -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write skeleton**

Create `examples/multi_machine_attn_zimage.py`:

```python
"""Standalone multi-machine Z-Image attention example.

Coordinator + worker over TCP. Outputs-only wire. SLALOM-verifies q/k/v/o
linears; recomputes the non-linear core (per-head RMSNorm + complex-cis
RoPE + softmax + attn matmuls) on the coordinator.

See docs/superpowers/specs/2026-05-14-multi-machine-attn-zimage-design.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _multi_machine_common as mmc                                     # noqa: E402
from _multi_machine_common import (                                     # noqa: E402
    DTYPE_FP16, DTYPE_FP32,
    MSG_CLOSE, MSG_FORWARD_DONE, MSG_FORWARD_REQ,
    MSG_LOAD_ACK, MSG_LOAD_REQ, MSG_TENSOR,
    RoundMetricsBase, SLALOM_K, S_GENERATOR_SEED, WireProtocolError,
    _DTYPE_NAME, _DTYPE_SIZE, _NAME_TO_DTYPE, _TORCH_DTYPE,
    apply_rotary_emb_zimage, default_slalom_threshold, format_summary,
    launch_loopback_worker, cleanup_loopback_worker, make_s, pack_tensor,
    pick_free_port, precompute_s_tilde, precompute_zimage_freqs_cis,
    recv_msg, rmsnorm_cpu, send_msg, slalom_verify, slalom_verify_safe,
    unpack_tensor, wait_port,
)


OP_Q = 1
OP_K = 2
OP_V = 3
OP_O = 4
_OP_NAME = {OP_Q: "q", OP_K: "k", OP_V: "v", OP_O: "o"}

ZIMAGE_QK_NORM_EPS = 1e-6


@dataclass
class AttnZimageConfig:
    dim: int = 1536
    heads: int = 12
    head_dim: int = 128
    batch: int = 2
    seq: int = 1024
    qk_norm: str = "rms"           # "rms" or "none"
    rope_theta: float = 10000.0
    wire_dtype: int = DTYPE_FP16
    weight_seed: int = 0xC0FFEE

    def __post_init__(self):
        assert self.dim % self.heads == 0
        assert self.dim // self.heads == self.head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even (RoPE)"
        assert self.qk_norm in ("rms", "none")


def make_zimage_attn_weights(
    cfg: AttnZimageConfig, dtype: torch.dtype, device: str | torch.device,
) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear,
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Returns (q_proj, k_proj, v_proj, o_proj, norm_q_w, norm_k_w).

    norm_q_w / norm_k_w are fp32 1-D tensors (head_dim,) — None when
    qk_norm == 'none'.
    """
    def _lin(in_d: int, out_d: int, offset: int) -> nn.Linear:
        gen = torch.Generator(device="cpu").manual_seed(cfg.weight_seed + offset)
        m = nn.Linear(in_d, out_d, bias=False)
        with torch.no_grad():
            m.weight.normal_(0.0, 0.02, generator=gen)
        return m.to(device=device, dtype=dtype)

    q = _lin(cfg.dim, cfg.heads * cfg.head_dim, offset=0)
    k = _lin(cfg.dim, cfg.heads * cfg.head_dim, offset=1)
    v = _lin(cfg.dim, cfg.heads * cfg.head_dim, offset=2)
    o = _lin(cfg.heads * cfg.head_dim, cfg.dim, offset=3)

    if cfg.qk_norm == "rms":
        gen_q = torch.Generator(device="cpu").manual_seed(cfg.weight_seed + 4)
        gen_k = torch.Generator(device="cpu").manual_seed(cfg.weight_seed + 5)
        nq = torch.empty(cfg.head_dim, dtype=torch.float32)
        nk = torch.empty(cfg.head_dim, dtype=torch.float32)
        nq.normal_(0.0, 0.02, generator=gen_q)
        nk.normal_(0.0, 0.02, generator=gen_k)
    else:
        nq = nk = None
    return q, k, v, o, nq, nk


def compute_zimage_attn_forward(
    x: torch.Tensor,
    q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear,
    norm_q_w: Optional[torch.Tensor], norm_k_w: Optional[torch.Tensor],
    freqs_cis: torch.Tensor, cfg: AttnZimageConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs zimage-attn forward. Returns (q_raw, k_raw, v_raw, output)."""
    B, S = cfg.batch, cfg.seq
    q_raw = q_proj(x)
    k_raw = k_proj(x)
    v_raw = v_proj(x)

    q = q_raw.unflatten(-1, (cfg.heads, cfg.head_dim))   # (B,S,H,D)
    k = k_raw.unflatten(-1, (cfg.heads, cfg.head_dim))
    v = v_raw.unflatten(-1, (cfg.heads, cfg.head_dim))

    if norm_q_w is not None:
        q = rmsnorm_cpu(q, norm_q_w.to(q.device), ZIMAGE_QK_NORM_EPS,
                        scale_offset=0.0)
    if norm_k_w is not None:
        k = rmsnorm_cpu(k, norm_k_w.to(k.device), ZIMAGE_QK_NORM_EPS,
                        scale_offset=0.0)

    q = apply_rotary_emb_zimage(q, freqs_cis.to(q.device))
    k = apply_rotary_emb_zimage(k, freqs_cis.to(k.device))

    q_t = q.permute(0, 2, 1, 3)                           # (B,H,S,D)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    scores = q_t @ k_t.transpose(2, 3) * (cfg.head_dim ** -0.5)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ v_t).permute(0, 2, 1, 3).flatten(2, 3)
    output = o_proj(attn_out)
    return q_raw, k_raw, v_raw, output
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_attn_zimage.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_zimage.py tests/test_multi_machine_attn_zimage.py
git commit -m "feat(mm-attn-zimage): skeleton, config, weights with QK RMSNorm, forward compute"
```

---

### Task C2: Worker, Coordinator, message bodies, fault injection

**Files:**
- Modify: `examples/multi_machine_attn_zimage.py` (append)
- Modify: `tests/test_multi_machine_attn_zimage.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_attn_zimage.py`:

```python
import socket


def test_loopback_round_passes_with_no_fault():
    m = _load()
    cfg = m.AttnZimageConfig(dim=32, heads=4, head_dim=8, batch=2, seq=8,
                             qk_norm="rms", wire_dtype=m.DTYPE_FP32,
                             weight_seed=11)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", pipeline=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K, pipeline=False)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is True
    assert rm.bytes_recv == rm.bytes_recv_predicted


def test_loopback_round_detects_drop_qk_norm_fault():
    m = _load()
    cfg = m.AttnZimageConfig(dim=32, heads=4, head_dim=8, batch=2, seq=8,
                             qk_norm="rms", wire_dtype=m.DTYPE_FP32,
                             weight_seed=11)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="drop_qk_norm", pipeline=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K, pipeline=False)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is False
    assert rm.mse[m.OP_O] > 1e-2
```

- [ ] **Step 2: Run failing tests**

Expected: FAIL — `Worker` and `Coordinator` not yet defined.

- [ ] **Step 3: Append message bodies, Worker, Coordinator**

Append to `examples/multi_machine_attn_zimage.py`:

```python

# ── Wire message bodies ─────────────────────────────────────────────
# LOAD_REQ body: <I I I I B B>
#   dim, heads, head_dim, weight_seed, qk_norm_id (0=none/1=rms), dtype_id
# rope_theta is sent as a separate u32 (theta_milli = int(theta*1000))
_LOAD_REQ_FMT = "<IIIIIBB"


def pack_load_req(dim: int, heads: int, head_dim: int,
                  weight_seed: int, rope_theta_e3: int,
                  qk_norm_id: int, dtype_id: int) -> bytes:
    return struct.pack(_LOAD_REQ_FMT, dim, heads, head_dim,
                       weight_seed, rope_theta_e3, qk_norm_id, dtype_id)


def unpack_load_req(body: bytes) -> dict:
    d, h, hd, seed, theta_e3, qkn, dtype = struct.unpack(_LOAD_REQ_FMT, body)
    return {"dim": d, "heads": h, "head_dim": hd, "weight_seed": seed,
            "rope_theta": theta_e3 / 1000.0,
            "qk_norm": "rms" if qkn == 1 else "none",
            "dtype_id": dtype}


def pack_load_ack(status: int) -> bytes:
    return struct.pack("<B", status)


def unpack_load_ack(body: bytes) -> dict:
    return {"status": struct.unpack("<B", body)[0]}


_FWD_REQ_FMT = "<QIII"


def pack_forward_req(request_id: int, input_seed: int,
                     batch: int, seq: int) -> bytes:
    return struct.pack(_FWD_REQ_FMT, request_id, input_seed, batch, seq)


def unpack_forward_req(body: bytes) -> dict:
    rid, seed, b, s = struct.unpack(_FWD_REQ_FMT, body)
    return {"request_id": rid, "input_seed": seed, "batch": b, "seq": s}


def pack_forward_done(request_id: int, gpu_t_ms: float) -> bytes:
    return struct.pack("<Qd", request_id, gpu_t_ms)


def unpack_forward_done(body: bytes) -> dict:
    rid, t = struct.unpack("<Qd", body)
    return {"request_id": rid, "gpu_forward_t_ms": t}


# ── Worker ──────────────────────────────────────────────────────────
class Worker:
    def __init__(self, bind_host: str, bind_port: int, device: str,
                 inject_fault: str = "none", pipeline: bool = False,
                 quiet: bool = False):
        self.bind_host = bind_host; self.bind_port = bind_port
        self.device = torch.device(device)
        self.inject_fault = inject_fault; self.pipeline = pipeline
        self.quiet = quiet
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.norm_q_w = self.norm_k_w = None
        self.dim = self.heads = self.head_dim = 0
        self.rope_theta = 10000.0
        self.qk_norm = "rms"
        self.wire_dtype_id = DTYPE_FP16
        self.compute_dtype = torch.float16
        self.cfg: Optional[AttnZimageConfig] = None
        self._round_count = 0

    def _log(self, msg: str) -> None:
        if self.quiet:
            return
        print(f"[zi-worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def serve_once(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.bind_port))
        srv.listen(8)
        self._log(f"listening on {self.bind_host}:{self.bind_port}  "
                  f"device={self.device}  fault={self.inject_fault}")
        try:
            while True:
                sock, addr = srv.accept()
                self._log(f"connection from {addr[0]}:{addr[1]}")
                try:
                    handled = self._serve_session(sock)
                finally:
                    sock.close()
                if handled:
                    return
        finally:
            srv.close()

    def _serve_session(self, sock) -> bool:
        handled = False
        while True:
            try:
                mt, body = recv_msg(sock)
            except (ConnectionError, OSError):
                return handled
            handled = True
            if mt == MSG_LOAD_REQ:
                self._handle_load(sock, unpack_load_req(body))
            elif mt == MSG_FORWARD_REQ:
                self._handle_forward(sock, unpack_forward_req(body))
            elif mt == MSG_CLOSE:
                return handled
            else:
                raise WireProtocolError(f"unexpected msg_type {mt}")

    def _handle_load(self, sock, fields: dict) -> None:
        self.dim = fields["dim"]; self.heads = fields["heads"]
        self.head_dim = fields["head_dim"]
        self.rope_theta = fields["rope_theta"]
        self.qk_norm = fields["qk_norm"]
        self.wire_dtype_id = fields["dtype_id"]
        self.compute_dtype = _TORCH_DTYPE[self.wire_dtype_id]
        self.cfg = AttnZimageConfig(
            dim=self.dim, heads=self.heads, head_dim=self.head_dim,
            batch=1, seq=1, qk_norm=self.qk_norm, rope_theta=self.rope_theta,
            wire_dtype=self.wire_dtype_id, weight_seed=fields["weight_seed"])
        (self.q_proj, self.k_proj, self.v_proj, self.o_proj,
         self.norm_q_w, self.norm_k_w) = make_zimage_attn_weights(
            self.cfg, dtype=self.compute_dtype, device=self.device)
        send_msg(sock, MSG_LOAD_ACK, pack_load_ack(0))
        self._log(f"LOAD: dim={self.dim} heads={self.heads} "
                  f"head_dim={self.head_dim} qk_norm={self.qk_norm}")

    def _handle_forward(self, sock, fields: dict) -> None:
        rid = fields["request_id"]; seed = fields["input_seed"]
        B, S = fields["batch"], fields["seq"]
        cfg = AttnZimageConfig(
            dim=self.dim, heads=self.heads, head_dim=self.head_dim,
            batch=B, seq=S, qk_norm=self.qk_norm, rope_theta=self.rope_theta,
            wire_dtype=self.wire_dtype_id, weight_seed=self.cfg.weight_seed)
        freqs = precompute_zimage_freqs_cis(self.head_dim, S, theta=self.rope_theta)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        x_cpu = torch.randn(B, S, self.dim, dtype=torch.float32, generator=gen)
        x = x_cpu.to(device=self.device, dtype=self.compute_dtype)

        t0 = time.perf_counter()
        q_raw, k_raw, v_raw, output = compute_zimage_attn_forward(
            x, self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.norm_q_w, self.norm_k_w, freqs, cfg)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_t = (time.perf_counter() - t0) * 1000.0

        q_raw, k_raw, v_raw, output = self._apply_fault(
            q_raw, k_raw, v_raw, output, x, freqs, cfg)

        if self.pipeline:
            self._send_pipelined(sock, rid, q_raw, k_raw, v_raw, output, gpu_t)
        else:
            for tag, t in ((OP_Q, q_raw), (OP_K, k_raw),
                           (OP_V, v_raw), (OP_O, output)):
                send_msg(sock, MSG_TENSOR,
                         pack_tensor(rid, tag, t, self.wire_dtype_id))
            send_msg(sock, MSG_FORWARD_DONE, pack_forward_done(rid, gpu_t))
        self._round_count += 1
        fault = f"  [INJECTED FAULT: {self.inject_fault}]" if self.inject_fault != "none" else ""
        self._log(f"round #{self._round_count} (req {rid}): "
                  f"x[{B},{S},{self.dim}] forward {gpu_t:.2f} ms{fault}")

    def _send_pipelined(self, sock, rid, q, k, v, o, gpu_t) -> None:
        send_q: queue.Queue = queue.Queue()
        sender_exc: list = []

        def _sender():
            try:
                while True:
                    item = send_q.get()
                    if item is None:
                        return
                    mtype, body = item
                    send_msg(sock, mtype, body)
            except Exception as e:
                sender_exc.append(e)

        t = threading.Thread(target=_sender, daemon=True)
        t.start()
        for tag, ten in ((OP_Q, q), (OP_K, k), (OP_V, v), (OP_O, o)):
            send_q.put((MSG_TENSOR, pack_tensor(rid, tag, ten, self.wire_dtype_id)))
        send_q.put((MSG_FORWARD_DONE, pack_forward_done(rid, gpu_t)))
        send_q.put(None)
        t.join()
        if sender_exc:
            raise sender_exc[0]

    def _apply_fault(self, q, k, v, o, x, freqs, cfg):
        if self.inject_fault == "none":
            return q, k, v, o
        if self.inject_fault == "flip_v":
            return q, k, -v, o
        if self.inject_fault == "scale_o":
            return q, k, v, o * 1.01
        if self.inject_fault == "drop_softmax":
            B, S = cfg.batch, cfg.seq
            qh = q.unflatten(-1, (cfg.heads, cfg.head_dim))
            kh = k.unflatten(-1, (cfg.heads, cfg.head_dim))
            vh = v.unflatten(-1, (cfg.heads, cfg.head_dim))
            if self.norm_q_w is not None:
                qh = rmsnorm_cpu(qh, self.norm_q_w.to(qh.device), ZIMAGE_QK_NORM_EPS)
            if self.norm_k_w is not None:
                kh = rmsnorm_cpu(kh, self.norm_k_w.to(kh.device), ZIMAGE_QK_NORM_EPS)
            qh = apply_rotary_emb_zimage(qh, freqs.to(qh.device))
            kh = apply_rotary_emb_zimage(kh, freqs.to(kh.device))
            qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
            scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
            attn_out = (scores @ vt).permute(0, 2, 1, 3).flatten(2, 3)
            return q, k, v, self.o_proj(attn_out)
        if self.inject_fault == "drop_rope":
            B, S = cfg.batch, cfg.seq
            qh = q.unflatten(-1, (cfg.heads, cfg.head_dim))
            kh = k.unflatten(-1, (cfg.heads, cfg.head_dim))
            vh = v.unflatten(-1, (cfg.heads, cfg.head_dim))
            if self.norm_q_w is not None:
                qh = rmsnorm_cpu(qh, self.norm_q_w.to(qh.device), ZIMAGE_QK_NORM_EPS)
            if self.norm_k_w is not None:
                kh = rmsnorm_cpu(kh, self.norm_k_w.to(kh.device), ZIMAGE_QK_NORM_EPS)
            # NO RoPE
            qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
            scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            attn_out = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
            return q, k, v, self.o_proj(attn_out)
        if self.inject_fault == "drop_qk_norm":
            B, S = cfg.batch, cfg.seq
            qh = q.unflatten(-1, (cfg.heads, cfg.head_dim))
            kh = k.unflatten(-1, (cfg.heads, cfg.head_dim))
            vh = v.unflatten(-1, (cfg.heads, cfg.head_dim))
            # NO QK RMSNorm
            qh = apply_rotary_emb_zimage(qh, freqs.to(qh.device))
            kh = apply_rotary_emb_zimage(kh, freqs.to(kh.device))
            qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
            scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            attn_out = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
            return q, k, v, self.o_proj(attn_out)
        raise ValueError(f"unknown inject_fault: {self.inject_fault}")


# ── Coordinator ─────────────────────────────────────────────────────
class Coordinator:
    def __init__(self, host: str, port: int, config: AttnZimageConfig,
                 threshold: float, k: int = SLALOM_K, pipeline: bool = False,
                 o_threshold: Optional[float] = None):
        self.host = host; self.port = port; self.config = config
        self.threshold = threshold
        self.o_threshold = o_threshold if o_threshold is not None else threshold
        self.k = k; self.pipeline = pipeline
        (self.q_proj, self.k_proj, self.v_proj, self.o_proj,
         self.norm_q_w, self.norm_k_w) = make_zimage_attn_weights(
            config, dtype=torch.float32, device="cpu")
        out_qkv = config.heads * config.head_dim
        self.s_q = make_s(out_qkv, k, seed=S_GENERATOR_SEED + 1)
        self.s_k = make_s(out_qkv, k, seed=S_GENERATOR_SEED + 2)
        self.s_v = make_s(out_qkv, k, seed=S_GENERATOR_SEED + 3)
        self.s_o = make_s(config.dim, k, seed=S_GENERATOR_SEED + 4)
        self.s_tilde_q = precompute_s_tilde(self.q_proj.weight, self.s_q)
        self.s_tilde_k = precompute_s_tilde(self.k_proj.weight, self.s_k)
        self.s_tilde_v = precompute_s_tilde(self.v_proj.weight, self.s_v)
        self.s_tilde_o = precompute_s_tilde(self.o_proj.weight, self.s_o)
        self._freqs_cache: dict[int, torch.Tensor] = {}
        self.sock: Optional[socket.socket] = None
        self.pool = ThreadPoolExecutor(max_workers=4)

    def _get_freqs(self, seq: int) -> torch.Tensor:
        if seq not in self._freqs_cache:
            self._freqs_cache[seq] = precompute_zimage_freqs_cis(
                self.config.head_dim, seq, theta=self.config.rope_theta)
        return self._freqs_cache[seq]

    def connect_and_load(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        body = pack_load_req(
            dim=self.config.dim, heads=self.config.heads,
            head_dim=self.config.head_dim, weight_seed=self.config.weight_seed,
            rope_theta_e3=int(self.config.rope_theta * 1000),
            qk_norm_id=(1 if self.config.qk_norm == "rms" else 0),
            dtype_id=self.config.wire_dtype)
        send_msg(self.sock, MSG_LOAD_REQ, body)
        mt, ack = recv_msg(self.sock)
        if mt != MSG_LOAD_ACK:
            raise WireProtocolError(f"expected LOAD_ACK, got {mt}")
        if unpack_load_ack(ack)["status"] != 0:
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

    def reproduce_input_cpu(self, input_seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(input_seed)
        return torch.randn(self.config.batch, self.config.seq, self.config.dim,
                           dtype=torch.float32, generator=gen)

    def predicted_recv_bytes(self) -> int:
        frame_hdr = 8
        tensor_body_hdr = struct.calcsize("<QHBB") + 4 * 3
        done_body = struct.calcsize("<Qd")
        ds = _DTYPE_SIZE[self.config.wire_dtype]
        cfg = self.config
        bytes_qkv = cfg.batch * cfg.seq * cfg.heads * cfg.head_dim * ds
        bytes_o = cfg.batch * cfg.seq * cfg.dim * ds
        return (4 * (frame_hdr + tensor_body_hdr) + (frame_hdr + done_body)
                + 3 * bytes_qkv + bytes_o)

    def _expected_shape(self, op_tag: int) -> tuple:
        cfg = self.config
        if op_tag in (OP_Q, OP_K, OP_V):
            return (cfg.batch, cfg.seq, cfg.heads * cfg.head_dim)
        if op_tag == OP_O:
            return (cfg.batch, cfg.seq, cfg.dim)
        raise WireProtocolError(f"unknown op_tag {op_tag}")

    def _recompute_attn_cpu(self, q_cpu, k_cpu, v_cpu) -> torch.Tensor:
        cfg = self.config
        freqs = self._get_freqs(cfg.seq)
        q = q_cpu.unflatten(-1, (cfg.heads, cfg.head_dim))
        k = k_cpu.unflatten(-1, (cfg.heads, cfg.head_dim))
        v = v_cpu.unflatten(-1, (cfg.heads, cfg.head_dim))
        if self.norm_q_w is not None:
            q = rmsnorm_cpu(q, self.norm_q_w, ZIMAGE_QK_NORM_EPS)
        if self.norm_k_w is not None:
            k = rmsnorm_cpu(k, self.norm_k_w, ZIMAGE_QK_NORM_EPS)
        q = apply_rotary_emb_zimage(q, freqs)
        k = apply_rotary_emb_zimage(k, freqs)
        q_t = q.permute(0, 2, 1, 3); k_t = k.permute(0, 2, 1, 3); v_t = v.permute(0, 2, 1, 3)
        scores = q_t @ k_t.transpose(2, 3) * (cfg.head_dim ** -0.5)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        return (probs @ v_t).permute(0, 2, 1, 3).flatten(2, 3)

    @staticmethod
    def _timed(fn, *args):
        t0 = time.perf_counter()
        mse = fn(*args)
        return mse, (time.perf_counter() - t0) * 1000.0

    def run_round(self, request_id: int, input_seed: int) -> RoundMetricsBase:
        assert self.sock is not None
        rm = RoundMetricsBase(request_id=request_id)
        rm.bytes_recv_predicted = self.predicted_recv_bytes()
        x_cpu = self.reproduce_input_cpu(input_seed)
        t_start = time.perf_counter()
        rm.bytes_sent = send_msg(
            self.sock, MSG_FORWARD_REQ,
            pack_forward_req(request_id, input_seed,
                             self.config.batch, self.config.seq))
        t_wire_start = time.perf_counter()
        bytes_recv = 0; acts = {}; gpu_t = 0.0; done = False
        while not done or len(acts) < 4:
            mt, body = recv_msg(self.sock)
            bytes_recv += 8 + len(body)
            if mt == MSG_TENSOR:
                d = unpack_tensor(body)
                op = d["op_tag"]; t = d["tensor"]
                if tuple(t.shape) != self._expected_shape(op):
                    raise WireProtocolError(
                        f"op_tag={op}: expected {self._expected_shape(op)}, "
                        f"got {tuple(t.shape)}")
                rm.recv_tensors[op] = {"shape": list(t.shape),
                                       "dtype": _DTYPE_NAME[d["dtype_id"]],
                                       "bytes": 8 + len(body)}
                acts[op] = t.to(torch.float32)
            elif mt == MSG_FORWARD_DONE:
                gpu_t = unpack_forward_done(body)["gpu_forward_t_ms"]; done = True
            else:
                raise WireProtocolError(f"unexpected msg_type {mt}")
        rm.bytes_recv = bytes_recv
        rm.gpu_forward_t = gpu_t
        rm.wire_recv_t = (time.perf_counter() - t_wire_start) * 1000.0

        t_v = time.perf_counter()
        f_q = self.pool.submit(self._timed, slalom_verify_safe,
                               x_cpu, acts[OP_Q], self.s_q, self.s_tilde_q)
        f_k = self.pool.submit(self._timed, slalom_verify_safe,
                               x_cpu, acts[OP_K], self.s_k, self.s_tilde_k)
        f_v = self.pool.submit(self._timed, slalom_verify_safe,
                               x_cpu, acts[OP_V], self.s_v, self.s_tilde_v)
        rm.mse[OP_Q], rm.cpu_verify_per_op_t[OP_Q] = f_q.result()
        rm.mse[OP_K], rm.cpu_verify_per_op_t[OP_K] = f_k.result()
        rm.mse[OP_V], rm.cpu_verify_per_op_t[OP_V] = f_v.result()
        attn_out_cpu = self._recompute_attn_cpu(acts[OP_Q], acts[OP_K], acts[OP_V])
        rm.mse[OP_O], rm.cpu_verify_per_op_t[OP_O] = self._timed(
            slalom_verify_safe, attn_out_cpu, acts[OP_O], self.s_o, self.s_tilde_o)
        rm.cpu_verify_t = (time.perf_counter() - t_v) * 1000.0
        rm.end_to_end_t = (time.perf_counter() - t_start) * 1000.0
        rm.ok = (rm.mse[OP_Q] <= self.threshold and
                 rm.mse[OP_K] <= self.threshold and
                 rm.mse[OP_V] <= self.threshold and
                 rm.mse[OP_O] <= self.o_threshold)
        return rm

    def run_many(self, rounds: int, *, input_seed_start: int = 1_000_000) -> list:
        return [self.run_round(i, input_seed_start + i) for i in range(rounds)]
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_attn_zimage.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_zimage.py tests/test_multi_machine_attn_zimage.py
git commit -m "feat(mm-attn-zimage): Worker, Coordinator, fault injection (5 fault kinds)"
```

---

### Task C3: CLI / main / loopback subprocess test

**Files:**
- Modify: `examples/multi_machine_attn_zimage.py` (append)
- Modify: `tests/test_multi_machine_attn_zimage.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_attn_zimage.py`:

```python
import subprocess


def test_loopback_subprocess_smoke():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--dim", "32", "--heads", "4", "--head-dim", "8",
         "--batch", "2", "--seq", "8",
         "--wire-dtype", "fp32", "--rounds", "5", "--warmup", "1"],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "rounds passed     4 / 4" in out.stdout


def test_loopback_subprocess_drop_qk_norm_fault():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--dim", "32", "--heads", "4", "--head-dim", "8",
         "--batch", "2", "--seq", "8",
         "--wire-dtype", "fp32", "--rounds", "3", "--warmup", "0",
         "--inject-fault", "drop_qk_norm"],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0
    assert "rounds passed     0 / 3" in out.stdout
```

- [ ] **Step 2: Run failing tests**

Expected: FAIL — example has no `main()`.

- [ ] **Step 3: Append CLI / main**

Append to `examples/multi_machine_attn_zimage.py`:

```python

# ── CLI driver ──────────────────────────────────────────────────────
def run_coordinator(host: str, port: int, args) -> int:
    cfg = AttnZimageConfig(
        dim=args.dim, heads=args.heads, head_dim=args.head_dim,
        batch=args.batch, seq=args.seq, qk_norm=args.qk_norm,
        rope_theta=args.rope_theta, wire_dtype=_NAME_TO_DTYPE[args.wire_dtype],
        weight_seed=args.weight_seed)
    threshold = args.threshold
    if threshold is None:
        threshold = default_slalom_threshold(cfg.wire_dtype, cfg.dim)
    o_threshold = default_slalom_threshold(cfg.wire_dtype, cfg.dim, fp16_slope=4e-6)
    coord = Coordinator(host=host, port=port, config=cfg,
                        threshold=threshold, k=SLALOM_K,
                        pipeline=args.pipeline, o_threshold=o_threshold)
    try:
        coord.connect_and_load()
        rounds = coord.run_many(rounds=args.rounds)
    finally:
        coord.close()
    config_lines = [
        f"Attn:     Z-Image  dim={cfg.dim}  heads={cfg.heads}  "
        f"head_dim={cfg.head_dim}  qk_norm={cfg.qk_norm}",
        f"Shape:    batch={cfg.batch}  seq={cfg.seq}  "
        f"dtype={_DTYPE_NAME[cfg.wire_dtype]}",
        f"RoPE:     theta={cfg.rope_theta} (complex-cis)",
        f"Verify:   SLALOM  k={SLALOM_K}  thr_qkv={threshold:.1e}  "
        f"thr_o={o_threshold:.1e}",
        f"Pipeline: {'on' if args.pipeline else 'off'}",
    ]
    print(format_summary(
        rounds, warmup=args.warmup, pipelined=args.pipeline,
        op_names=_OP_NAME, config_lines=config_lines,
        link_gbps=args.link_gbps,
        title="Multi-Machine Z-Image Attention Example"))
    return 0


def launch_loopback(args) -> int:
    proc, port = launch_loopback_worker(
        __file__, extra_worker_argv=["--inject-fault", args.inject_fault]
                  + (["--pipeline"] if args.pipeline else []),
        device=args.device)
    try:
        wait_port(port, timeout=10.0)
        return run_coordinator("127.0.0.1", port, args)
    finally:
        cleanup_loopback_worker(proc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["loopback", "worker", "coordinator"],
                   default="loopback")
    p.add_argument("--bind", default="127.0.0.1:9102")
    p.add_argument("--worker-host", default="127.0.0.1")
    p.add_argument("--worker-port", type=int, default=9102)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--qk-norm", choices=["rms", "none"], default="rms")
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--wire-dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--weight-seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject-fault", default="none",
                   choices=["none", "flip_v", "scale_o", "drop_softmax",
                            "drop_rope", "drop_qk_norm"])
    p.add_argument("--link-gbps", type=float, default=None)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.role == "worker":
        host, port_s = args.bind.split(":")
        Worker(host, int(port_s), args.device, args.inject_fault,
               pipeline=args.pipeline, quiet=args.quiet).serve_once()
        return 0
    if args.role == "coordinator":
        return run_coordinator(args.worker_host, args.worker_port, args)
    if args.role == "loopback":
        return launch_loopback(args)
    raise ValueError(f"unknown role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_attn_zimage.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_attn_zimage.py tests/test_multi_machine_attn_zimage.py
git commit -m "feat(mm-attn-zimage): CLI, run_coordinator, launch_loopback, main entrypoint"
```

---

## Phase D — Mini-Zimage example (`examples/multi_machine_zimage.py`)

### Task D1: Skeleton, ZimageConfig, multi-block weight builder

**Files:**
- Create: `examples/multi_machine_zimage.py`
- Create: `tests/test_multi_machine_zimage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_machine_zimage.py`:

```python
"""Functional tests for examples/multi_machine_zimage.py."""
from __future__ import annotations

import importlib.util
import pathlib
import socket
import subprocess
import sys
import threading

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_zimage.py"


def _load():
    spec = importlib.util.spec_from_file_location("mmzimage", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmzimage"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_op_kind_constants():
    m = _load()
    assert m.OP_Q == 1 and m.OP_K == 2 and m.OP_V == 3 and m.OP_O == 4
    assert m.OP_W1 == 5 and m.OP_W3 == 6 and m.OP_W2 == 7


def test_pack_unpack_op_tag_with_block():
    m = _load()
    # block_idx << 4 | op_kind
    tag = m.make_op_tag(block_idx=5, op_kind=m.OP_W1)
    assert tag == (5 << 4) | m.OP_W1
    block, kind = m.split_op_tag(tag)
    assert block == 5 and kind == m.OP_W1


def test_zimage_config_defaults():
    m = _load()
    cfg = m.ZimageConfig()
    assert cfg.dim == 1536
    assert cfg.heads == 12
    assert cfg.head_dim == 128
    assert cfg.ffn_inter == 4096
    assert cfg.n_layers == 12


def test_make_block_weights_shape_and_count():
    m = _load()
    cfg = m.ZimageConfig(dim=32, heads=4, head_dim=8, ffn_inter=64,
                         n_layers=2, batch=1, seq=4)
    blocks = m.make_zimage_block_weights(cfg, dtype=torch.float32, device="cpu")
    assert len(blocks) == cfg.n_layers
    block = blocks[0]
    assert block.q_proj.weight.shape == (4 * 8, 32)
    assert block.o_proj.weight.shape == (32, 4 * 8)
    assert block.w1.weight.shape == (64, 32)
    assert block.w2.weight.shape == (32, 64)
    assert block.w3.weight.shape == (64, 32)
    assert block.attention_norm1.shape == (32,)
    assert block.ffn_norm1.shape == (32,)


def test_make_block_weights_per_block_unique():
    m = _load()
    cfg = m.ZimageConfig(dim=16, heads=2, head_dim=8, ffn_inter=32,
                         n_layers=2, batch=1, seq=2, weight_seed=1)
    blocks = m.make_zimage_block_weights(cfg, dtype=torch.float32, device="cpu")
    # Block 0 and block 1 must have DIFFERENT q_proj weights
    assert not torch.equal(blocks[0].q_proj.weight, blocks[1].q_proj.weight)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q tests/test_multi_machine_zimage.py -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write skeleton**

Create `examples/multi_machine_zimage.py`:

```python
"""Standalone multi-machine mini-Zimage example.

Stack of N (RMSNorm + zimage-attn + RMSNorm + SwiGLU) transformer blocks.
Coordinator + worker over TCP. Outputs-only wire. SLALOM-verifies every
linear in every block; recomputes RMSNorms / RoPE / softmax / silu / residuals
on the coordinator.

NOT a wrap of diffusers ZImageTransformer2DModel — see spec section 9.

See docs/superpowers/specs/2026-05-14-multi-machine-attn-zimage-design.md.
"""
from __future__ import annotations

import argparse
import pathlib
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _multi_machine_common as mmc                                     # noqa: E402
from _multi_machine_common import (                                     # noqa: E402
    DTYPE_FP16, DTYPE_FP32,
    MSG_CLOSE, MSG_FORWARD_DONE, MSG_FORWARD_REQ,
    MSG_LOAD_ACK, MSG_LOAD_REQ, MSG_TENSOR,
    RoundMetricsBase, SLALOM_K, S_GENERATOR_SEED, WireProtocolError,
    _DTYPE_NAME, _DTYPE_SIZE, _NAME_TO_DTYPE, _TORCH_DTYPE,
    apply_rotary_emb_zimage, default_slalom_threshold, format_summary,
    launch_loopback_worker, cleanup_loopback_worker, make_s, pack_tensor,
    pick_free_port, precompute_s_tilde, precompute_zimage_freqs_cis,
    recv_msg, rmsnorm_cpu, send_msg, slalom_verify, slalom_verify_safe,
    unpack_tensor, wait_port,
)


# ── Op tag namespace ────────────────────────────────────────────────
# Wire op_tag is uint16: (block_idx << 4) | op_kind. Up to 4096 blocks × 7 kinds.
OP_Q  = 1
OP_K  = 2
OP_V  = 3
OP_O  = 4
OP_W1 = 5
OP_W3 = 6
OP_W2 = 7

_OP_KIND_NAMES = {OP_Q: "q", OP_K: "k", OP_V: "v", OP_O: "o",
                  OP_W1: "w1", OP_W3: "w3", OP_W2: "w2"}


def make_op_tag(block_idx: int, op_kind: int) -> int:
    assert 0 <= block_idx < 4096, "block_idx must fit in 12 bits"
    assert 1 <= op_kind <= 7
    return (block_idx << 4) | op_kind


def split_op_tag(tag: int) -> tuple[int, int]:
    return (tag >> 4) & 0xFFF, tag & 0xF


def op_label(tag: int) -> str:
    b, k = split_op_tag(tag)
    return f"b{b}.{_OP_KIND_NAMES.get(k, 'op?')}"


ZIMAGE_QK_NORM_EPS = 1e-6
ZIMAGE_LAYER_NORM_EPS = 1e-6


# ── Config ──────────────────────────────────────────────────────────
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

    def __post_init__(self):
        assert self.dim % self.heads == 0
        assert self.dim // self.heads == self.head_dim
        assert self.head_dim % 2 == 0
        assert self.qk_norm in ("rms", "none")
        assert 1 <= self.n_layers < 4096


# ── Per-block weight container ──────────────────────────────────────
@dataclass
class BlockWeights:
    attention_norm1: torch.Tensor      # (dim,) fp32 or compute_dtype
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    norm_q: Optional[torch.Tensor]     # (head_dim,) or None
    norm_k: Optional[torch.Tensor]
    o_proj: nn.Linear
    ffn_norm1: torch.Tensor
    w1: nn.Linear
    w3: nn.Linear
    w2: nn.Linear


# ── Multi-block weight builder ──────────────────────────────────────
_BLOCK_STRIDE = 16   # per-block seed stride (offsets 0..10 used; 16 leaves headroom)


def make_zimage_block_weights(
    cfg: ZimageConfig, dtype: torch.dtype, device: str | torch.device,
) -> list[BlockWeights]:
    blocks: list[BlockWeights] = []
    for b in range(cfg.n_layers):
        base = cfg.weight_seed + b * _BLOCK_STRIDE

        def _lin(in_d: int, out_d: int, off: int) -> nn.Linear:
            gen = torch.Generator(device="cpu").manual_seed(base + off)
            m = nn.Linear(in_d, out_d, bias=False)
            with torch.no_grad():
                m.weight.normal_(0.0, 0.02, generator=gen)
            return m.to(device=device, dtype=dtype)

        def _norm(shape: int, off: int) -> torch.Tensor:
            gen = torch.Generator(device="cpu").manual_seed(base + off)
            t = torch.empty(shape, dtype=torch.float32)
            t.normal_(0.0, 0.02, generator=gen)
            return t

        bw = BlockWeights(
            attention_norm1=_norm(cfg.dim, 0),
            q_proj=_lin(cfg.dim, cfg.heads * cfg.head_dim, 1),
            k_proj=_lin(cfg.dim, cfg.heads * cfg.head_dim, 2),
            v_proj=_lin(cfg.dim, cfg.heads * cfg.head_dim, 3),
            norm_q=_norm(cfg.head_dim, 4) if cfg.qk_norm == "rms" else None,
            norm_k=_norm(cfg.head_dim, 5) if cfg.qk_norm == "rms" else None,
            o_proj=_lin(cfg.heads * cfg.head_dim, cfg.dim, 6),
            ffn_norm1=_norm(cfg.dim, 7),
            w1=_lin(cfg.dim, cfg.ffn_inter, 8),
            w3=_lin(cfg.dim, cfg.ffn_inter, 9),
            w2=_lin(cfg.ffn_inter, cfg.dim, 10),
        )
        blocks.append(bw)
    return blocks
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/test_multi_machine_zimage.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_zimage.py tests/test_multi_machine_zimage.py
git commit -m "feat(mm-zimage): skeleton, ZimageConfig, BlockWeights, multi-block weight builder"
```

---

### Task D2: Worker N-block forward compute (sequential send)

**Files:**
- Modify: `examples/multi_machine_zimage.py` (append)
- Modify: `tests/test_multi_machine_zimage.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_zimage.py`:

```python
def test_compute_zimage_block_forward_matches_reference():
    m = _load()
    cfg = m.ZimageConfig(dim=16, heads=2, head_dim=8, ffn_inter=32,
                         n_layers=2, batch=1, seq=4, weight_seed=42)
    blocks = m.make_zimage_block_weights(cfg, torch.float32, "cpu")
    freqs = m.precompute_zimage_freqs_cis(cfg.head_dim, cfg.seq, theta=cfg.rope_theta)

    torch.manual_seed(0)
    x_in = torch.randn(cfg.batch, cfg.seq, cfg.dim)

    # Run the worker compute
    block_outs, x_out = m.compute_zimage_stack_forward(x_in, blocks, freqs, cfg)
    assert len(block_outs) == cfg.n_layers
    # Each block_outs[b] is dict with keys OP_Q/OP_K/OP_V/OP_O/OP_W1/OP_W3/OP_W2
    keys = {m.OP_Q, m.OP_K, m.OP_V, m.OP_O, m.OP_W1, m.OP_W3, m.OP_W2}
    assert set(block_outs[0].keys()) == keys

    # Reference: re-run block by block using the same compute paths
    x = x_in
    for b in range(cfg.n_layers):
        bw = blocks[b]
        x_n = m.rmsnorm_cpu(x, bw.attention_norm1, m.ZIMAGE_LAYER_NORM_EPS)
        # attention sub-block
        q = bw.q_proj(x_n); k = bw.k_proj(x_n); v = bw.v_proj(x_n)
        qh = q.unflatten(-1, (cfg.heads, cfg.head_dim))
        kh = k.unflatten(-1, (cfg.heads, cfg.head_dim))
        vh = v.unflatten(-1, (cfg.heads, cfg.head_dim))
        if bw.norm_q is not None:
            qh = m.rmsnorm_cpu(qh, bw.norm_q, m.ZIMAGE_QK_NORM_EPS)
            kh = m.rmsnorm_cpu(kh, bw.norm_k, m.ZIMAGE_QK_NORM_EPS)
        qh = m.apply_rotary_emb_zimage(qh, freqs)
        kh = m.apply_rotary_emb_zimage(kh, freqs)
        qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
        scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn_out = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
        o = bw.o_proj(attn_out)
        x_after = x + o

        # FFN sub-block
        h = m.rmsnorm_cpu(x_after, bw.ffn_norm1, m.ZIMAGE_LAYER_NORM_EPS)
        w1o = bw.w1(h); w3o = bw.w3(h)
        gated = torch.nn.functional.silu(w1o) * w3o
        w2o = bw.w2(gated)
        x = x_after + w2o

    assert torch.allclose(x_out, x, atol=1e-4)
```

- [ ] **Step 2: Run failing test**

Run: `pytest -q tests/test_multi_machine_zimage.py -k stack_forward -v`
Expected: FAIL — `compute_zimage_stack_forward` not defined.

- [ ] **Step 3: Append worker compute**

Append to `examples/multi_machine_zimage.py`:

```python

# ── N-block forward (worker side) ───────────────────────────────────
def compute_zimage_block_forward(
    x_in: torch.Tensor, bw: BlockWeights, freqs_cis: torch.Tensor,
    cfg: ZimageConfig,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run one block forward. Returns (per-op-kind raw outputs, x_out_for_next_block)."""
    B, S = cfg.batch, cfg.seq

    # Attention sub-block
    x_norm = rmsnorm_cpu(x_in,
                         bw.attention_norm1.to(x_in.device),
                         ZIMAGE_LAYER_NORM_EPS).to(x_in.dtype)
    q_raw = bw.q_proj(x_norm)
    k_raw = bw.k_proj(x_norm)
    v_raw = bw.v_proj(x_norm)

    qh = q_raw.unflatten(-1, (cfg.heads, cfg.head_dim))
    kh = k_raw.unflatten(-1, (cfg.heads, cfg.head_dim))
    vh = v_raw.unflatten(-1, (cfg.heads, cfg.head_dim))
    if bw.norm_q is not None:
        qh = rmsnorm_cpu(qh, bw.norm_q.to(qh.device), ZIMAGE_QK_NORM_EPS).to(qh.dtype)
        kh = rmsnorm_cpu(kh, bw.norm_k.to(kh.device), ZIMAGE_QK_NORM_EPS).to(kh.dtype)
    qh = apply_rotary_emb_zimage(qh, freqs_cis.to(qh.device))
    kh = apply_rotary_emb_zimage(kh, freqs_cis.to(kh.device))
    qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
    scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
    o_raw = bw.o_proj(attn_out)
    x_after = x_in + o_raw

    # FFN sub-block
    h = rmsnorm_cpu(x_after,
                    bw.ffn_norm1.to(x_after.device),
                    ZIMAGE_LAYER_NORM_EPS).to(x_after.dtype)
    w1_raw = bw.w1(h)
    w3_raw = bw.w3(h)
    gated = F.silu(w1_raw) * w3_raw
    w2_raw = bw.w2(gated)
    x_out = x_after + w2_raw

    return {OP_Q: q_raw, OP_K: k_raw, OP_V: v_raw, OP_O: o_raw,
            OP_W1: w1_raw, OP_W3: w3_raw, OP_W2: w2_raw}, x_out


def compute_zimage_stack_forward(
    x_in: torch.Tensor, blocks: list[BlockWeights],
    freqs_cis: torch.Tensor, cfg: ZimageConfig,
) -> tuple[list[dict[int, torch.Tensor]], torch.Tensor]:
    """Run the full N-block forward. Returns (per-block raw-outputs list, final x)."""
    block_outs = []
    x = x_in
    for b in range(cfg.n_layers):
        outs, x = compute_zimage_block_forward(x, blocks[b], freqs_cis, cfg)
        block_outs.append(outs)
    return block_outs, x
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_zimage.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_zimage.py tests/test_multi_machine_zimage.py
git commit -m "feat(mm-zimage): per-block and N-block stack forward compute"
```

---

### Task D3: Worker class (LOAD/FORWARD/CLOSE) with sequential and streaming send

**Files:**
- Modify: `examples/multi_machine_zimage.py` (append)
- Modify: `tests/test_multi_machine_zimage.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_zimage.py`:

```python
def test_zimage_worker_load_round_trip():
    m = _load()
    cfg_kwargs = dict(dim=16, heads=2, head_dim=8, ffn_inter=32, n_layers=2,
                      batch=1, seq=4, weight_seed=7)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", fault_block=0, stream=False,
                      quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = m.pack_load_req(
        dim=cfg_kwargs["dim"], heads=cfg_kwargs["heads"],
        head_dim=cfg_kwargs["head_dim"], ffn_inter=cfg_kwargs["ffn_inter"],
        n_layers=cfg_kwargs["n_layers"],
        weight_seed=cfg_kwargs["weight_seed"],
        rope_theta_e3=int(10000.0 * 1000),
        qk_norm_id=1, dtype_id=m.DTYPE_FP32,
    )
    m.send_msg(sock, m.MSG_LOAD_REQ, body)
    mt, ack = m.recv_msg(sock)
    assert mt == m.MSG_LOAD_ACK
    assert m.unpack_load_ack(ack)["status"] == 0
    m.send_msg(sock, m.MSG_CLOSE, b"")
    sock.close()
    t.join(2.0)
    assert worker.n_layers == cfg_kwargs["n_layers"]
    assert len(worker.blocks) == cfg_kwargs["n_layers"]
```

- [ ] **Step 2: Run failing test**

Expected: FAIL — `Worker` not defined.

- [ ] **Step 3: Append message bodies and Worker**

Append to `examples/multi_machine_zimage.py`:

```python

# ── Wire message bodies ─────────────────────────────────────────────
# LOAD_REQ body: <I I I I I I I B B>
#   dim, heads, head_dim, ffn_inter, n_layers, weight_seed, rope_theta_e3,
#   qk_norm_id, dtype_id
_LOAD_REQ_FMT = "<IIIIIIIBB"


def pack_load_req(dim: int, heads: int, head_dim: int, ffn_inter: int,
                  n_layers: int, weight_seed: int, rope_theta_e3: int,
                  qk_norm_id: int, dtype_id: int) -> bytes:
    return struct.pack(_LOAD_REQ_FMT, dim, heads, head_dim, ffn_inter,
                       n_layers, weight_seed, rope_theta_e3,
                       qk_norm_id, dtype_id)


def unpack_load_req(body: bytes) -> dict:
    (d, h, hd, fi, nl, seed, theta_e3, qkn, dtype) = struct.unpack(
        _LOAD_REQ_FMT, body)
    return {"dim": d, "heads": h, "head_dim": hd, "ffn_inter": fi,
            "n_layers": nl, "weight_seed": seed,
            "rope_theta": theta_e3 / 1000.0,
            "qk_norm": "rms" if qkn == 1 else "none",
            "dtype_id": dtype}


def pack_load_ack(status: int) -> bytes:
    return struct.pack("<B", status)


def unpack_load_ack(body: bytes) -> dict:
    return {"status": struct.unpack("<B", body)[0]}


_FWD_REQ_FMT = "<QIII"


def pack_forward_req(request_id: int, input_seed: int,
                     batch: int, seq: int) -> bytes:
    return struct.pack(_FWD_REQ_FMT, request_id, input_seed, batch, seq)


def unpack_forward_req(body: bytes) -> dict:
    rid, seed, b, s = struct.unpack(_FWD_REQ_FMT, body)
    return {"request_id": rid, "input_seed": seed, "batch": b, "seq": s}


def pack_forward_done(request_id: int, gpu_t_ms: float) -> bytes:
    return struct.pack("<Qd", request_id, gpu_t_ms)


def unpack_forward_done(body: bytes) -> dict:
    rid, t = struct.unpack("<Qd", body)
    return {"request_id": rid, "gpu_forward_t_ms": t}


# ── Worker ──────────────────────────────────────────────────────────
class Worker:
    def __init__(self, bind_host: str, bind_port: int, device: str,
                 inject_fault: str = "none", fault_block: int = 0,
                 stream: bool = True, quiet: bool = False):
        self.bind_host = bind_host; self.bind_port = bind_port
        self.device = torch.device(device)
        self.inject_fault = inject_fault
        self.fault_block = fault_block
        self.stream = stream; self.quiet = quiet
        self.blocks: list[BlockWeights] = []
        self.dim = self.heads = self.head_dim = self.ffn_inter = 0
        self.n_layers = 0
        self.qk_norm = "rms"; self.rope_theta = 10000.0
        self.wire_dtype_id = DTYPE_FP16
        self.compute_dtype = torch.float16
        self.cfg: Optional[ZimageConfig] = None
        self._round_count = 0

    def _log(self, msg: str) -> None:
        if self.quiet:
            return
        print(f"[zi-worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def serve_once(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.bind_port))
        srv.listen(8)
        self._log(f"listening on {self.bind_host}:{self.bind_port}  "
                  f"device={self.device}  fault={self.inject_fault}@b{self.fault_block}")
        try:
            while True:
                sock, addr = srv.accept()
                self._log(f"connection from {addr[0]}:{addr[1]}")
                try:
                    handled = self._serve_session(sock)
                finally:
                    sock.close()
                if handled:
                    return
        finally:
            srv.close()

    def _serve_session(self, sock) -> bool:
        handled = False
        while True:
            try:
                mt, body = recv_msg(sock)
            except (ConnectionError, OSError):
                return handled
            handled = True
            if mt == MSG_LOAD_REQ:
                self._handle_load(sock, unpack_load_req(body))
            elif mt == MSG_FORWARD_REQ:
                self._handle_forward(sock, unpack_forward_req(body))
            elif mt == MSG_CLOSE:
                return handled
            else:
                raise WireProtocolError(f"unexpected msg_type {mt}")

    def _handle_load(self, sock, fields: dict) -> None:
        self.dim = fields["dim"]; self.heads = fields["heads"]
        self.head_dim = fields["head_dim"]; self.ffn_inter = fields["ffn_inter"]
        self.n_layers = fields["n_layers"]
        self.qk_norm = fields["qk_norm"]; self.rope_theta = fields["rope_theta"]
        self.wire_dtype_id = fields["dtype_id"]
        self.compute_dtype = _TORCH_DTYPE[self.wire_dtype_id]
        self.cfg = ZimageConfig(
            dim=self.dim, heads=self.heads, head_dim=self.head_dim,
            ffn_inter=self.ffn_inter, n_layers=self.n_layers,
            batch=1, seq=1, qk_norm=self.qk_norm, rope_theta=self.rope_theta,
            wire_dtype=self.wire_dtype_id, weight_seed=fields["weight_seed"])
        self.blocks = make_zimage_block_weights(
            self.cfg, dtype=self.compute_dtype, device=self.device)
        send_msg(sock, MSG_LOAD_ACK, pack_load_ack(0))
        self._log(f"LOAD: dim={self.dim} n_layers={self.n_layers} "
                  f"ffn_inter={self.ffn_inter}")

    def _handle_forward(self, sock, fields: dict) -> None:
        rid = fields["request_id"]; seed = fields["input_seed"]
        B, S = fields["batch"], fields["seq"]
        cfg = ZimageConfig(
            dim=self.dim, heads=self.heads, head_dim=self.head_dim,
            ffn_inter=self.ffn_inter, n_layers=self.n_layers,
            batch=B, seq=S, qk_norm=self.qk_norm, rope_theta=self.rope_theta,
            wire_dtype=self.wire_dtype_id, weight_seed=self.cfg.weight_seed)
        freqs = precompute_zimage_freqs_cis(self.head_dim, S, theta=self.rope_theta)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        x_cpu = torch.randn(B, S, self.dim, dtype=torch.float32, generator=gen)
        x = x_cpu.to(device=self.device, dtype=self.compute_dtype)

        # Optional streaming sender thread
        send_q: Optional[queue.Queue] = None
        sender_t: Optional[threading.Thread] = None
        sender_exc: list = []
        if self.stream:
            send_q = queue.Queue()

            def _sender():
                try:
                    while True:
                        item = send_q.get()
                        if item is None:
                            return
                        mtype, body = item
                        send_msg(sock, mtype, body)
                except Exception as e:
                    sender_exc.append(e)

            sender_t = threading.Thread(target=_sender, daemon=True)
            sender_t.start()

        t0 = time.perf_counter()
        x_in = x
        for b in range(self.n_layers):
            outs, x_in = compute_zimage_block_forward(
                x_in, self.blocks[b], freqs, cfg)
            outs = self._apply_fault_for_block(b, outs, x_in, freqs, cfg)
            for kind in (OP_Q, OP_K, OP_V, OP_O, OP_W1, OP_W3, OP_W2):
                tag = make_op_tag(b, kind)
                body = pack_tensor(rid, tag, outs[kind], self.wire_dtype_id)
                if self.stream:
                    send_q.put((MSG_TENSOR, body))
                else:
                    send_msg(sock, MSG_TENSOR, body)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_t = (time.perf_counter() - t0) * 1000.0

        if self.stream:
            send_q.put((MSG_FORWARD_DONE, pack_forward_done(rid, gpu_t)))
            send_q.put(None)
            sender_t.join()
            if sender_exc:
                raise sender_exc[0]
        else:
            send_msg(sock, MSG_FORWARD_DONE, pack_forward_done(rid, gpu_t))

        self._round_count += 1
        fault = (f"  [INJECTED FAULT: {self.inject_fault}@b{self.fault_block}]"
                 if self.inject_fault != "none" else "")
        mode = "streamed" if self.stream else "sequential"
        self._log(f"round #{self._round_count} (req {rid}): "
                  f"x[{B},{S},{self.dim}] N={self.n_layers} forward "
                  f"{gpu_t:.2f} ms ({mode}){fault}")

    def _apply_fault_for_block(self, b: int, outs: dict, x_in_for_next, freqs, cfg
                                ) -> dict:
        if self.inject_fault == "none" or b != self.fault_block:
            return outs
        if self.inject_fault == "flip_v":
            outs[OP_V] = -outs[OP_V]
        elif self.inject_fault == "scale_o":
            outs[OP_O] = outs[OP_O] * 1.01
        elif self.inject_fault == "scale_w2":
            outs[OP_W2] = outs[OP_W2] * 1.01
        elif self.inject_fault == "flip_w1":
            outs[OP_W1] = -outs[OP_W1]
        elif self.inject_fault == "drop_silu":
            # We can't easily reconstruct here without the gated input;
            # mark outs[OP_W2] as bw.w2(w1*w3) — caught by w2 SLALOM.
            bw = self.blocks[b]
            gated_bad = outs[OP_W1] * outs[OP_W3]                 # missing silu
            outs[OP_W2] = bw.w2(gated_bad)
        else:
            raise ValueError(f"unknown inject_fault: {self.inject_fault}")
        return outs
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_zimage.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_zimage.py tests/test_multi_machine_zimage.py
git commit -m "feat(mm-zimage): Worker class with streaming sender and per-block fault injection"
```

---

### Task D4: Coordinator with chained block verify; CLI; integration tests

**Files:**
- Modify: `examples/multi_machine_zimage.py` (append)
- Modify: `tests/test_multi_machine_zimage.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_machine_zimage.py`:

```python
def test_zimage_loopback_round_passes_with_no_fault():
    m = _load()
    cfg = m.ZimageConfig(dim=16, heads=2, head_dim=8, ffn_inter=32,
                         n_layers=2, batch=1, seq=4,
                         wire_dtype=m.DTYPE_FP32, weight_seed=11)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", stream=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is True
    assert rm.bytes_recv == rm.bytes_recv_predicted
    # Per-block per-op MSE all under threshold
    for b in range(cfg.n_layers):
        for kind in (m.OP_Q, m.OP_K, m.OP_V, m.OP_O, m.OP_W1, m.OP_W3, m.OP_W2):
            tag = m.make_op_tag(b, kind)
            assert rm.mse[tag] < 1e-3, f"block {b} kind {kind} mse={rm.mse[tag]}"


def test_zimage_loopback_subprocess_smoke():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--dim", "16", "--heads", "2", "--head-dim", "8",
         "--ffn-inter", "32", "--n-layers", "2",
         "--batch", "1", "--seq", "4",
         "--wire-dtype", "fp32", "--rounds", "5", "--warmup", "1"],
        capture_output=True, timeout=120, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "rounds passed     4 / 4" in out.stdout


def test_zimage_loopback_subprocess_block_propagation():
    """A scale_o fault at block 0 must also poison block 1's MSE."""
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--dim", "16", "--heads", "2", "--head-dim", "8",
         "--ffn-inter", "32", "--n-layers", "2",
         "--batch", "1", "--seq", "4",
         "--wire-dtype", "fp32", "--rounds", "1", "--warmup", "0",
         "--inject-fault", "scale_o", "--fault-block", "0"],
        capture_output=True, timeout=120, text=True,
    )
    assert out.returncode == 0
    assert "rounds passed     0 / 1" in out.stdout
    # Both block 0 (where fault was injected) and block 1 (where verified
    # x_in_for_next diverges from worker's x_in_for_next) must show high MSE
    # on at least one op_kind. We verify by checking the printed mse summary
    # contains TWO different "p95" entries with values > 1e-2.
    high_mse = sum(1 for line in out.stdout.splitlines()
                   if "mse[" in line and "p95" in line
                   and float(line.rsplit()[-1]) > 1e-2)
    assert high_mse >= 2, f"expected ≥2 high-mse ops, got {high_mse}\n{out.stdout}"
```

- [ ] **Step 2: Run failing tests**

Expected: FAIL — `Coordinator` not defined.

- [ ] **Step 3: Append Coordinator + CLI + main**

Append to `examples/multi_machine_zimage.py`:

```python

# ── Coordinator ─────────────────────────────────────────────────────
class Coordinator:
    def __init__(self, host: str, port: int, config: ZimageConfig,
                 threshold: float, k: int = SLALOM_K,
                 o_threshold: Optional[float] = None,
                 max_workers: int = 4):
        self.host = host; self.port = port; self.config = config
        self.threshold = threshold
        self.o_threshold = o_threshold if o_threshold is not None else threshold
        self.k = k
        self.blocks = make_zimage_block_weights(config, dtype=torch.float32,
                                                device="cpu")
        # Per-block, per-op SLALOM keys
        out_qkv = config.heads * config.head_dim
        self.s_q  = [make_s(out_qkv, k, S_GENERATOR_SEED + 1 + b * 16)
                     for b in range(config.n_layers)]
        self.s_k  = [make_s(out_qkv, k, S_GENERATOR_SEED + 2 + b * 16)
                     for b in range(config.n_layers)]
        self.s_v  = [make_s(out_qkv, k, S_GENERATOR_SEED + 3 + b * 16)
                     for b in range(config.n_layers)]
        self.s_o  = [make_s(config.dim, k, S_GENERATOR_SEED + 4 + b * 16)
                     for b in range(config.n_layers)]
        self.s_w1 = [make_s(config.ffn_inter, k, S_GENERATOR_SEED + 5 + b * 16)
                     for b in range(config.n_layers)]
        self.s_w3 = [make_s(config.ffn_inter, k, S_GENERATOR_SEED + 6 + b * 16)
                     for b in range(config.n_layers)]
        self.s_w2 = [make_s(config.dim, k, S_GENERATOR_SEED + 7 + b * 16)
                     for b in range(config.n_layers)]
        # Precompute s_tilde per block per op
        self.s_tilde_q  = [precompute_s_tilde(self.blocks[b].q_proj.weight, self.s_q[b])
                           for b in range(config.n_layers)]
        self.s_tilde_k  = [precompute_s_tilde(self.blocks[b].k_proj.weight, self.s_k[b])
                           for b in range(config.n_layers)]
        self.s_tilde_v  = [precompute_s_tilde(self.blocks[b].v_proj.weight, self.s_v[b])
                           for b in range(config.n_layers)]
        self.s_tilde_o  = [precompute_s_tilde(self.blocks[b].o_proj.weight, self.s_o[b])
                           for b in range(config.n_layers)]
        self.s_tilde_w1 = [precompute_s_tilde(self.blocks[b].w1.weight, self.s_w1[b])
                           for b in range(config.n_layers)]
        self.s_tilde_w3 = [precompute_s_tilde(self.blocks[b].w3.weight, self.s_w3[b])
                           for b in range(config.n_layers)]
        self.s_tilde_w2 = [precompute_s_tilde(self.blocks[b].w2.weight, self.s_w2[b])
                           for b in range(config.n_layers)]
        self._freqs_cache: dict[int, torch.Tensor] = {}
        self.sock: Optional[socket.socket] = None
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _get_freqs(self, seq: int) -> torch.Tensor:
        if seq not in self._freqs_cache:
            self._freqs_cache[seq] = precompute_zimage_freqs_cis(
                self.config.head_dim, seq, theta=self.config.rope_theta)
        return self._freqs_cache[seq]

    def connect_and_load(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        body = pack_load_req(
            dim=self.config.dim, heads=self.config.heads,
            head_dim=self.config.head_dim, ffn_inter=self.config.ffn_inter,
            n_layers=self.config.n_layers, weight_seed=self.config.weight_seed,
            rope_theta_e3=int(self.config.rope_theta * 1000),
            qk_norm_id=(1 if self.config.qk_norm == "rms" else 0),
            dtype_id=self.config.wire_dtype)
        send_msg(self.sock, MSG_LOAD_REQ, body)
        mt, ack = recv_msg(self.sock)
        if mt != MSG_LOAD_ACK:
            raise WireProtocolError(f"expected LOAD_ACK, got {mt}")
        if unpack_load_ack(ack)["status"] != 0:
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

    def reproduce_input_cpu(self, input_seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(input_seed)
        return torch.randn(self.config.batch, self.config.seq, self.config.dim,
                           dtype=torch.float32, generator=gen)

    def predicted_recv_bytes(self) -> int:
        frame_hdr = 8
        tensor_body_hdr = struct.calcsize("<QHBB") + 4 * 3
        done_body = struct.calcsize("<Qd")
        ds = _DTYPE_SIZE[self.config.wire_dtype]
        cfg = self.config
        bytes_qkv = cfg.batch * cfg.seq * cfg.heads * cfg.head_dim * ds
        bytes_o   = cfg.batch * cfg.seq * cfg.dim * ds
        bytes_w13 = cfg.batch * cfg.seq * cfg.ffn_inter * ds
        bytes_w2  = bytes_o
        per_block = (3 * bytes_qkv + bytes_o + 2 * bytes_w13 + bytes_w2)
        per_block_frames = 7 * (frame_hdr + tensor_body_hdr)
        return (cfg.n_layers * (per_block + per_block_frames)
                + (frame_hdr + done_body))

    def _expected_shape(self, op_kind: int) -> tuple:
        cfg = self.config
        if op_kind in (OP_Q, OP_K, OP_V):
            return (cfg.batch, cfg.seq, cfg.heads * cfg.head_dim)
        if op_kind == OP_O:
            return (cfg.batch, cfg.seq, cfg.dim)
        if op_kind in (OP_W1, OP_W3):
            return (cfg.batch, cfg.seq, cfg.ffn_inter)
        if op_kind == OP_W2:
            return (cfg.batch, cfg.seq, cfg.dim)
        raise WireProtocolError(f"unknown op_kind {op_kind}")

    def _verify_block(
        self, b: int, x_in_b: torch.Tensor, ops: dict[int, torch.Tensor],
    ) -> tuple[dict[int, float], torch.Tensor]:
        """Verify block b's seven tensors using x_in_b as the chain input.

        Returns (per-op-kind mse dict, x_out_for_next_block).
        """
        cfg = self.config
        bw = self.blocks[b]
        freqs = self._get_freqs(cfg.seq)

        # Attention sub-block input
        x_norm = rmsnorm_cpu(x_in_b, bw.attention_norm1, ZIMAGE_LAYER_NORM_EPS)
        mse: dict[int, float] = {}
        mse[OP_Q] = slalom_verify_safe(x_norm, ops[OP_Q], self.s_q[b], self.s_tilde_q[b])
        mse[OP_K] = slalom_verify_safe(x_norm, ops[OP_K], self.s_k[b], self.s_tilde_k[b])
        mse[OP_V] = slalom_verify_safe(x_norm, ops[OP_V], self.s_v[b], self.s_tilde_v[b])

        # Recompute attn → CPU input for o_proj
        q = ops[OP_Q].unflatten(-1, (cfg.heads, cfg.head_dim))
        k = ops[OP_K].unflatten(-1, (cfg.heads, cfg.head_dim))
        v = ops[OP_V].unflatten(-1, (cfg.heads, cfg.head_dim))
        if bw.norm_q is not None:
            q = rmsnorm_cpu(q, bw.norm_q, ZIMAGE_QK_NORM_EPS)
            k = rmsnorm_cpu(k, bw.norm_k, ZIMAGE_QK_NORM_EPS)
        q = apply_rotary_emb_zimage(q, freqs)
        k = apply_rotary_emb_zimage(k, freqs)
        qt = q.permute(0, 2, 1, 3); kt = k.permute(0, 2, 1, 3); vt = v.permute(0, 2, 1, 3)
        scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_out_cpu = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
        mse[OP_O] = slalom_verify_safe(attn_out_cpu, ops[OP_O],
                                        self.s_o[b], self.s_tilde_o[b])
        x_after = x_in_b + ops[OP_O]

        # FFN sub-block
        h = rmsnorm_cpu(x_after, bw.ffn_norm1, ZIMAGE_LAYER_NORM_EPS)
        mse[OP_W1] = slalom_verify_safe(h, ops[OP_W1], self.s_w1[b], self.s_tilde_w1[b])
        mse[OP_W3] = slalom_verify_safe(h, ops[OP_W3], self.s_w3[b], self.s_tilde_w3[b])
        gated_cpu = F.silu(ops[OP_W1]) * ops[OP_W3]
        mse[OP_W2] = slalom_verify_safe(gated_cpu, ops[OP_W2],
                                         self.s_w2[b], self.s_tilde_w2[b])
        x_out = x_after + ops[OP_W2]
        return mse, x_out

    def run_round(self, request_id: int, input_seed: int) -> RoundMetricsBase:
        assert self.sock is not None
        rm = RoundMetricsBase(request_id=request_id)
        rm.bytes_recv_predicted = self.predicted_recv_bytes()
        x_cpu = self.reproduce_input_cpu(input_seed)
        t_start = time.perf_counter()
        rm.bytes_sent = send_msg(
            self.sock, MSG_FORWARD_REQ,
            pack_forward_req(request_id, input_seed,
                             self.config.batch, self.config.seq))

        # Receive all 7N tensors + DONE; collect by block
        t_wire_start = time.perf_counter()
        bytes_recv = 0
        per_block_acts: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.config.n_layers)]
        # As soon as block b's seven kinds are present AND the previous block's
        # x_out is ready, kick off block b's verify in the pool.
        block_x_in_futures: list[Optional[Future]] = [None] * self.config.n_layers
        block_mse_results: list[Optional[dict]] = [None] * self.config.n_layers

        def _kick_block(b: int):
            prev = block_x_in_futures[b - 1] if b > 0 else None
            x_in_b = prev.result()[1] if prev is not None else x_cpu

            def _task():
                mse, x_out = self._verify_block(b, x_in_b, per_block_acts[b])
                block_mse_results[b] = mse
                return mse, x_out

            block_x_in_futures[b] = self.pool.submit(_task)

        gpu_t = 0.0
        done = False
        while not done:
            mt, body = recv_msg(self.sock)
            bytes_recv += 8 + len(body)
            if mt == MSG_TENSOR:
                d = unpack_tensor(body)
                tag = d["op_tag"]; t = d["tensor"]
                b, kind = split_op_tag(tag)
                if not (0 <= b < self.config.n_layers and kind in _OP_KIND_NAMES):
                    raise WireProtocolError(f"unknown op_tag {tag} (b={b} kind={kind})")
                expected = self._expected_shape(kind)
                if tuple(t.shape) != expected:
                    raise WireProtocolError(
                        f"block {b} kind {kind}: expected {expected}, "
                        f"got {tuple(t.shape)}")
                rm.recv_tensors[tag] = {"shape": list(t.shape),
                                        "dtype": _DTYPE_NAME[d["dtype_id"]],
                                        "bytes": 8 + len(body)}
                per_block_acts[b][kind] = t.to(torch.float32)
                if (len(per_block_acts[b]) == 7 and block_x_in_futures[b] is None):
                    # Need block b-1's task to be in-flight or done before kicking b.
                    if b == 0 or block_x_in_futures[b - 1] is not None:
                        _kick_block(b)
            elif mt == MSG_FORWARD_DONE:
                gpu_t = unpack_forward_done(body)["gpu_forward_t_ms"]
                done = True
            else:
                raise WireProtocolError(f"unexpected msg_type {mt}")

        rm.bytes_recv = bytes_recv
        rm.gpu_forward_t = gpu_t
        rm.wire_recv_t = (time.perf_counter() - t_wire_start) * 1000.0

        # Catch up: kick any remaining un-kicked blocks (defensive — should be none
        # if streaming order matched expectations, but handles non-streaming sends).
        for b in range(self.config.n_layers):
            if block_x_in_futures[b] is None:
                _kick_block(b)

        # Wait for all block verifies, fold MSE into rm.mse keyed by full op_tag
        t_v_start = time.perf_counter()
        for b in range(self.config.n_layers):
            block_x_in_futures[b].result()  # ensure done
            for kind, val in block_mse_results[b].items():
                rm.mse[make_op_tag(b, kind)] = val
        rm.cpu_verify_t = (time.perf_counter() - t_v_start) * 1000.0
        rm.end_to_end_t = (time.perf_counter() - t_start) * 1000.0
        rm.ok = all(v <= self.threshold for v in rm.mse.values())
        return rm

    def run_many(self, rounds: int, *, input_seed_start: int = 1_000_000) -> list:
        return [self.run_round(i, input_seed_start + i) for i in range(rounds)]


# ── CLI ─────────────────────────────────────────────────────────────
def run_coordinator(host: str, port: int, args) -> int:
    cfg = ZimageConfig(
        dim=args.dim, heads=args.heads, head_dim=args.head_dim,
        ffn_inter=args.ffn_inter, n_layers=args.n_layers,
        batch=args.batch, seq=args.seq, qk_norm=args.qk_norm,
        rope_theta=args.rope_theta,
        wire_dtype=_NAME_TO_DTYPE[args.wire_dtype],
        weight_seed=args.weight_seed)
    threshold = args.threshold
    if threshold is None:
        threshold = default_slalom_threshold(cfg.wire_dtype, cfg.dim, fp16_slope=6e-6)
    coord = Coordinator(host=host, port=port, config=cfg,
                        threshold=threshold, k=SLALOM_K)
    try:
        coord.connect_and_load()
        rounds = coord.run_many(rounds=args.rounds)
    finally:
        coord.close()
    config_lines = [
        f"Mini-Zimage: dim={cfg.dim}  heads={cfg.heads}  head_dim={cfg.head_dim}  "
        f"ffn_inter={cfg.ffn_inter}  n_layers={cfg.n_layers}",
        f"Shape:    batch={cfg.batch}  seq={cfg.seq}  "
        f"dtype={_DTYPE_NAME[cfg.wire_dtype]}",
        f"QK norm:  {cfg.qk_norm}   RoPE theta={cfg.rope_theta}",
        f"Verify:   SLALOM  k={SLALOM_K}  thr={threshold:.1e}",
    ]
    op_names = {make_op_tag(b, kind): f"b{b}.{name}"
                for b in range(cfg.n_layers)
                for kind, name in _OP_KIND_NAMES.items()}
    print(format_summary(
        rounds, warmup=args.warmup, pipelined=False,
        op_names=op_names, config_lines=config_lines,
        link_gbps=args.link_gbps,
        title="Multi-Machine Mini-Zimage Example"))
    return 0


def launch_loopback(args) -> int:
    extra = ["--inject-fault", args.inject_fault,
             "--fault-block", str(args.fault_block)]
    if args.no_stream:
        extra.append("--no-stream")
    proc, port = launch_loopback_worker(
        __file__, extra_worker_argv=extra, device=args.device)
    try:
        wait_port(port, timeout=10.0)
        return run_coordinator("127.0.0.1", port, args)
    finally:
        cleanup_loopback_worker(proc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["loopback", "worker", "coordinator"],
                   default="loopback")
    p.add_argument("--bind", default="127.0.0.1:9103")
    p.add_argument("--worker-host", default="127.0.0.1")
    p.add_argument("--worker-port", type=int, default=9103)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--ffn-inter", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--qk-norm", choices=["rms", "none"], default="rms")
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--wire-dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--weight-seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject-fault", default="none",
                   choices=["none", "flip_v", "scale_o", "scale_w2",
                            "flip_w1", "drop_silu"])
    p.add_argument("--fault-block", type=int, default=0)
    p.add_argument("--no-stream", action="store_true",
                   help="disable worker streaming sender (sequential send)")
    p.add_argument("--link-gbps", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.role == "worker":
        host, port_s = args.bind.split(":")
        Worker(host, int(port_s), args.device, args.inject_fault,
               fault_block=args.fault_block,
               stream=not args.no_stream, quiet=args.quiet).serve_once()
        return 0
    if args.role == "coordinator":
        return run_coordinator(args.worker_host, args.worker_port, args)
    if args.role == "loopback":
        return launch_loopback(args)
    raise ValueError(f"unknown role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_multi_machine_zimage.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/multi_machine_zimage.py tests/test_multi_machine_zimage.py
git commit -m "feat(mm-zimage): Coordinator with chained block verify, CLI, integration tests"
```

---

## Phase E — Cross-cutting determinism & fault-margin tests

### Task E1: Parametrized cross-example invariants

**Files:**
- Modify: each of `tests/test_multi_machine_attn_llama.py`, `tests/test_multi_machine_attn_zimage.py`, `tests/test_multi_machine_zimage.py` (append two parametrized tests each)

- [ ] **Step 1: Append determinism + fault-margin tests to each example test file**

Append this block to **all three** test files (`tests/test_multi_machine_attn_llama.py`, `tests/test_multi_machine_attn_zimage.py`, `tests/test_multi_machine_zimage.py`). Adjust the `BASE_ARGS` list at the top of each block to match the example's CLI flags.

For `tests/test_multi_machine_attn_llama.py`:

```python
import json
import tempfile

BASE_ARGS_LLAMA = [
    "--role", "loopback", "--device", "cpu",
    "--hidden", "32", "--heads", "4", "--kv-heads", "4", "--head-dim", "8",
    "--batch", "2", "--seq", "8",
    "--wire-dtype", "fp32", "--rounds", "3", "--warmup", "0",
]


def _run_with_json(args: list[str]) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)] + args + ["--json-report", path],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    with open(path) as f:
        return json.load(f)


def test_llama_determinism_two_runs_identical_mse():
    a = _run_with_json(BASE_ARGS_LLAMA + ["--weight-seed", "0x1234"])
    b = _run_with_json(BASE_ARGS_LLAMA + ["--weight-seed", "0x1234"])
    a_mse = [(r["request_id"], r["mse"]) for r in a["per_round"]]
    b_mse = [(r["request_id"], r["mse"]) for r in b["per_round"]]
    assert a_mse == b_mse, "MSE must be deterministic for fixed seed"


@pytest.mark.parametrize("fault", ["flip_v", "scale_o", "drop_softmax", "drop_rope"])
def test_llama_fault_margin_at_least_10x_threshold(fault):
    rep = _run_with_json(BASE_ARGS_LLAMA + ["--inject-fault", fault])
    # Find the highest MSE across all rounds and ops
    max_mse = max(max(r["mse"].values()) for r in rep["per_round"])
    assert max_mse > 1e-2, f"fault {fault}: max mse {max_mse} not ≥10× 1e-3 threshold"
```

For `tests/test_multi_machine_attn_zimage.py` use the same structure but with `BASE_ARGS_ZIMAGE`:

```python
BASE_ARGS_ZIMAGE = [
    "--role", "loopback", "--device", "cpu",
    "--dim", "32", "--heads", "4", "--head-dim", "8",
    "--batch", "2", "--seq", "8",
    "--wire-dtype", "fp32", "--rounds", "3", "--warmup", "0",
]
# (Run the example in --role coordinator + --json-report path; identical structure.)
# Repeat _run_with_json (rename if needed) and the two test functions.

# faults to parametrize: ["flip_v", "scale_o", "drop_softmax", "drop_rope", "drop_qk_norm"]
```

For `tests/test_multi_machine_zimage.py`:

```python
BASE_ARGS_MINIZIMAGE = [
    "--role", "loopback", "--device", "cpu",
    "--dim", "16", "--heads", "2", "--head-dim", "8",
    "--ffn-inter", "32", "--n-layers", "2",
    "--batch", "1", "--seq", "4",
    "--wire-dtype", "fp32", "--rounds", "2", "--warmup", "0",
]
# faults: ["flip_v", "scale_o", "scale_w2", "flip_w1", "drop_silu"]
# Same _run_with_json + two test functions structure.
```

**Engineer note:** the JSON-report flag was added to the Llama example only (Task B5). For C and D, also add `--json-report PATH` to their CLI argparsers with the same write-out behavior — see B5 for the exact code, mirror it. If the JSON write-out is missing in those examples, add it as a small follow-up edit before running these E1 tests. (Each example's `run_coordinator` ends with the same `pathlib.Path(args.json_report).write_text(...)` block.)

- [ ] **Step 2: Run all tests**

Run: `pytest -q tests/test_multi_machine_*.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_multi_machine_attn_llama.py tests/test_multi_machine_attn_zimage.py tests/test_multi_machine_zimage.py examples/multi_machine_attn_zimage.py examples/multi_machine_zimage.py
git commit -m "test(mm-cross): determinism + fault-margin invariants for all three examples"
```

---

## Self-Review Checklist (post-implementation)

Run from repo root:

- [ ] All four examples' loopback smoke tests pass: `pytest -q tests/test_multi_machine_*.py`
- [ ] Existing FFN tests still pass (untouched): `pytest -q tests/test_multi_machine_ffn_example.py`
- [ ] Each example runs as a script: `python examples/multi_machine_attn_llama.py --device cpu --hidden 32 --heads 4 --kv-heads 4 --head-dim 8 --batch 2 --seq 8 --wire-dtype fp32 --rounds 5 --warmup 1` (and analogous for the other two — mini-zimage uses `--n-layers 2 --dim 16 --heads 2 --head-dim 8 --ffn-inter 32 --batch 1 --seq 4`).
- [ ] No `MSG_ACTIVATION` references in new files (must use `MSG_TENSOR` from common).
- [ ] No imports from `verified_core` or `verified_diffusers` in any of the four new example files (examples must be self-contained, with the only shared dep being `_multi_machine_common`).
- [ ] `examples/multi_machine_ffn.py` byte-identical to its pre-change state.

---

## Notes for the engineer

- **Why no JSON-report flag in C and D's Step 3 blocks?** The Llama example (Task B5) is the canonical one; mirror its JSON-write block into `run_coordinator` of zimage-attn and mini-zimage examples. If you forget, the E1 cross-cutting tests will fail with "FileNotFoundError" — that's your reminder.
- **Why a separate `_BLOCK_STRIDE = 16` constant in mini-zimage?** Per-block weight seeds use offsets 0..10 (11 distinct values), but stride 16 leaves headroom for new ops without breaking determinism on existing seeds.
- **Why does the mini-zimage Coordinator use `Future.result()` recursively in `_kick_block`?** Block `b`'s verify needs the verified `x_in` from block `b-1`. Threading through futures lets the receive loop continue accepting block `b+1`'s tensors while block `b-1` is still being verified — verify and wire-recv overlap.
- **Pipeline for attn examples is symmetric with FFN.** Worker threads packing+queuing, coordinator submits SLALOM as tensors arrive. The `--pipeline` flag is wired but the loopback test does not require it; it's measured via the `--pipeline` runs in the upstream `run_*.sh` perf scripts (out of scope here, follow-up).
- **What to do if a fault test passes but with margin < 10×?** Re-run with a different `--input-seed-start`; it's possible a particular input happens to have low forward-error sensitivity. If still <10×, raise the multiplier in the fault, or tighten the test assertion to ≥3× for that one fault and document why.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-14-multi-machine-attn-zimage.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
