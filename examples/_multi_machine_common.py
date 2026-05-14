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
