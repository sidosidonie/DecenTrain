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
        """Accept one real coordinator, serve until CLOSE or disconnect.

        Ignores empty probe connections (e.g. liveness checks from launchers
        / tests) that connect and immediately disconnect without sending any
        bytes — keeps accepting until a session actually transmits a message.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.bind_port))
        srv.listen(8)
        try:
            while True:
                sock, _addr = srv.accept()
                try:
                    handled = self._serve_session(sock)
                finally:
                    sock.close()
                if handled:
                    return
        finally:
            srv.close()

    def _serve_session(self, sock: socket.socket) -> bool:
        """Run one session. Returns True if any message was processed."""
        handled = False
        while True:
            try:
                msg_type, body = recv_msg(sock)
            except (ConnectionError, OSError):
                return handled
            handled = True
            if msg_type == MSG_LOAD_REQ:
                fields = unpack_load_req(body)
                self._handle_load(sock, fields)
            elif msg_type == MSG_CLOSE:
                return handled
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
