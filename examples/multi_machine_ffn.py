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

    def run_many(self, rounds: int, *, input_seed_start: int = 1_000_000) -> list[RoundMetrics]:
        return [self.run_round(i, input_seed_start + i) for i in range(rounds)]


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
