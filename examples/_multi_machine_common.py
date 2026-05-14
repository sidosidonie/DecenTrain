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
# u16 op_tag (not u8): mini-zimage encodes (block_idx<<4 | op_kind),
# supporting up to 4096 blocks × 16 op kinds.
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

    Accepts freqs_cis of shape (S, D/2) or (B, S, D/2).
    Mirrors verified_diffusers/zimage/attention.py:30-43.
    """
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Normalise freqs_cis to (*, S, 1, D/2) so it broadcasts with (B, S, H, D/2)
    fc = freqs_cis.unsqueeze(-2)                           # (S, 1, D/2) or (B, S, 1, D/2)
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
