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
_OP_NAME = {OP_W1: "y1 (w1)", OP_W3: "y3 (w3)", OP_W2: "y2 (w2)"}

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


def slalom_verify_safe(x, y, s, s_tilde) -> float:
    """Like slalom_verify but returns inf if y has any NaN/Inf.

    Without this guard, a worker returning NaN-poisoned activations
    would slip through (NaN > threshold is False, so ok would stay True).
    """
    if not torch.isfinite(y).all():
        return float("inf")
    return slalom_verify(x, y, s, s_tilde)


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
                 inject_fault: str = "none", pipeline: bool = False,
                 quiet: bool = False):
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.device = torch.device(device)
        self.inject_fault = inject_fault
        self.pipeline = pipeline
        self.quiet = quiet
        self.w1: Optional[nn.Linear] = None
        self.w2: Optional[nn.Linear] = None
        self.w3: Optional[nn.Linear] = None
        self.hidden = 0
        self.inter = 0
        self.wire_dtype_id = DTYPE_FP16
        self.compute_dtype: torch.dtype = torch.float16
        self._round_count = 0

    def _log(self, msg: str) -> None:
        """Progress line on stdout so it's visible whether the worker was
        launched directly (run_gpu_worker.sh) or as a loopback subprocess
        (whose stdout is inherited; only its stderr gets captured)."""
        if self.quiet:
            return
        print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _gpu_mem_str(self) -> str:
        if self.device.type != "cuda":
            return ""
        alloc = torch.cuda.memory_allocated(self.device) / 1e6
        return f", GPU mem {alloc:.0f} MB"

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
        dev_extra = (f" ({torch.cuda.get_device_name(self.device)})"
                     if self.device.type == "cuda" else "")
        self._log(f"listening on {self.bind_host}:{self.bind_port}  "
                  f"device={self.device}{dev_extra}  fault={self.inject_fault}  "
                  f"pipeline={self.pipeline}")
        self._log("waiting for a coordinator to connect…")
        try:
            while True:
                sock, _addr = srv.accept()
                self._log(f"connection from {_addr[0]}:{_addr[1]}")
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
                self._log("coordinator sent CLOSE")
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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        send_msg(sock, MSG_LOAD_ACK, pack_load_ack(0))
        self._log(f"LOAD: SwiGLU hidden={self.hidden} inter={self.inter} "
                  f"wire={_DTYPE_NAME[self.wire_dtype_id]} → weights on "
                  f"{self.device}{self._gpu_mem_str()}; ready for FORWARD requests")

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

        if self.pipeline:
            self._send_pipelined(sock, request_id, y1, y3, y2, gpu_t_ms)
        else:
            send_msg(sock, MSG_ACTIVATION,
                     pack_activation(request_id, OP_W1, y1, self.wire_dtype_id))
            send_msg(sock, MSG_ACTIVATION,
                     pack_activation(request_id, OP_W3, y3, self.wire_dtype_id))
            send_msg(sock, MSG_ACTIVATION,
                     pack_activation(request_id, OP_W2, y2, self.wire_dtype_id))
            send_msg(sock, MSG_FORWARD_DONE,
                     pack_forward_done(request_id, gpu_t_ms))

        self._round_count += 1
        sent_mb = ((y1.numel() + y3.numel() + y2.numel())
                   * _DTYPE_SIZE[self.wire_dtype_id]) / 1e6
        fault_note = f"  [INJECTED FAULT: {self.inject_fault}]" if self.inject_fault != "none" else ""
        send_mode = "pipelined" if self.pipeline else "sequential"
        self._log(f"round #{self._round_count} (req {request_id}): "
                  f"x[{batch},{seq},{self.hidden}] → forward {gpu_t_ms:.2f} ms on "
                  f"{self.device.type}, sent y1/y3{tuple(y1.shape)} y2{tuple(y2.shape)} "
                  f"= {sent_mb:.2f} MB ({send_mode}){fault_note}")

    def _send_pipelined(self, sock, request_id, y1, y3, y2, gpu_t_ms) -> None:
        """Pipelined send: a background thread pushes each activation over TCP
        while this thread continues D2H + packs the next one. The transmission
        of y1 overlaps with the D2H/packing of y3, y2 — a real win on slow
        links (negligible on loopback, but it demonstrates the mechanism)."""
        send_q: "queue.Queue" = queue.Queue()
        sender_exc: list = []

        def _sender():
            try:
                while True:
                    item = send_q.get()
                    if item is None:
                        return
                    mtype, body = item
                    send_msg(sock, mtype, body)
            except Exception as e:  # propagate to main thread
                sender_exc.append(e)

        sender = threading.Thread(target=_sender, daemon=True)
        sender.start()
        # pack_activation does the D2H synchronously; queueing hands the bytes
        # to the sender thread, freeing this thread to pack the next one.
        send_q.put((MSG_ACTIVATION, pack_activation(request_id, OP_W1, y1, self.wire_dtype_id)))
        send_q.put((MSG_ACTIVATION, pack_activation(request_id, OP_W3, y3, self.wire_dtype_id)))
        send_q.put((MSG_ACTIVATION, pack_activation(request_id, OP_W2, y2, self.wire_dtype_id)))
        send_q.put((MSG_FORWARD_DONE, pack_forward_done(request_id, gpu_t_ms)))
        send_q.put(None)
        sender.join()
        if sender_exc:
            raise sender_exc[0]

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
    # Per-tensor wire record for the activations the worker sent back:
    #   op_tag -> {"shape": [...], "dtype": "fp16", "bytes": <incl. msg header>}
    recv_tensors: dict = field(default_factory=dict)
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

    # wire_recv_t measures (request-sent → last-byte-received). The worker's
    # GPU forward happens during this window, so the actual on-wire portion
    # is wire_recv_t minus gpu_forward_t. Subtract so the three phases sum
    # to ~100% of e2e instead of double-counting GPU time.
    e2e = rm.end_to_end_t if rm.end_to_end_t > 0 else 1e-9
    wire_only_t = max(0.0, rm.wire_recv_t - rm.gpu_forward_t)
    return {
        "wire_mbps": wire_mbps,
        "gpu_gflops": gpu_gflops,
        "verify_gflops": verify_gflops,
        "gpu_pct": rm.gpu_forward_t / e2e,
        "wire_pct": wire_only_t / e2e,
        "verify_pct": rm.cpu_verify_t / e2e,
        "sum_pct": (rm.gpu_forward_t + wire_only_t + rm.cpu_verify_t) / e2e,
    }


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(xs, q))


def format_summary(rounds: list[RoundMetrics], cfg: FFNConfig, *,
                    warmup: int, k: int = SLALOM_K, pipelined: bool = False,
                    link_gbps: Optional[float] = None) -> str:
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

    # Per-tensor wire record. Shapes/dtypes are constant across rounds, so take
    # them from the first measured round that captured them.
    wire_tensors = next((r.recv_tensors for r in measured if r.recv_tensors), {})
    wire_tensor_lines = "".join(
        f"  {_OP_NAME.get(op, f'op{op}'):<10} "
        f"shape={str(tuple(info['shape'])):<18} "
        f"dtype={info['dtype']:<5} {info['bytes']/1e6:6.2f} MB\n"
        for op, info in sorted(wire_tensors.items())
    ) or "  (none captured)\n"

    # ── Wire estimate: ideal-vs-measured transfer for this payload size ──
    # "On-wire" time = the recv window minus the worker's forward (which runs
    # inside it), i.e. the time actually spent moving bytes back.
    payload_bytes = _mean([r.bytes_recv for r in measured])
    payload_bits = payload_bytes * 8.0
    wire_only_ms = _mean(
        [max(0.0, r.wire_recv_t - r.gpu_forward_t) for r in measured])
    eff_gbps = (payload_bits / (wire_only_ms / 1000.0) / 1e9) if wire_only_ms > 0 else 0.0
    ref_gbps = [1.0, 10.0, 25.0]
    if link_gbps and not any(abs(g - link_gbps) < 1e-9 for g in ref_gbps):
        ref_gbps = sorted(ref_gbps + [link_gbps])

    def _ideal_ms(gbps: float) -> float:
        return payload_bits / (gbps * 1e9) * 1000.0 if gbps > 0 else 0.0

    wire_est_lines = (
        f"  measured            {wire_only_ms:8.2f} ms  →  {eff_gbps:6.2f} "
        "Gbit/s effective  (recv window minus GPU forward)\n"
    )
    for g in ref_gbps:
        tag = "   ← --link-gbps" if link_gbps and abs(g - link_gbps) < 1e-9 else ""
        wire_est_lines += f"  @ {g:g} Gbit/s ideal  {_ideal_ms(g):8.2f} ms{tag}\n"
    if link_gbps:
        wire_est_lines += (
            f"  link efficiency     {eff_gbps / link_gbps * 100:8.1f} %   "
            f"of nominal {link_gbps:g} Gbit/s\n"
        )

    return (
        "=== Multi-Machine FFN Example: "
        f"{len(rounds)} rounds (warmup={warmup}) ===\n\n"
        "Config:\n"
        f"  FFN:      SwiGLU  hidden={cfg.hidden}  inter={cfg.inter}  "
        f"dtype={_DTYPE_NAME[cfg.wire_dtype]}  batch×seq={cfg.batch}×{cfg.seq}\n"
        f"  Verify:   SLALOM  k={k}\n"
        f"  Pipeline: {'on (send/verify overlap compute/recv)' if pipelined else 'off (phases sequential)'}\n\n"
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
        f"  sum_pct           {_mean([r['sum_pct'] for r in rates])*100:.1f}%"
        f"{'  (>100% = phases overlap, pipeline working)' if pipelined else '  (≈100% expected, sequential)'}\n\n"
        "Rate breakdown:\n"
        f"  Wire throughput   {_mean([r['wire_mbps'] for r in rates]):.1f} MB/s\n"
        f"  GPU compute       {_mean([r['gpu_gflops'] for r in rates])/1000.0:.2f} TFLOPS\n"
        f"  CPU SLALOM        {_mean([r['verify_gflops'] for r in rates]):.2f} GFLOPS\n\n"
        "Wire bytes (per round):\n"
        f"  Predicted         {bytes_pred_mb:.2f} MB\n"
        f"  Measured          {bytes_recv_mb:.2f} MB\n\n"
        "Wire tensors (per round, worker→coordinator):\n"
        f"{wire_tensor_lines}\n"
        f"Wire estimate (per round, {payload_bytes/1e6:.2f} MB payload):\n"
        f"{wire_est_lines}\n"
        "Verification:\n"
        f"  rounds passed     {passed} / {len(measured)}\n"
        f"  mse_w1 p95        {_percentile([r.mse_w1 for r in measured], 95):.2e}\n"
        f"  mse_w3 p95        {_percentile([r.mse_w3 for r in measured], 95):.2e}\n"
        f"  mse_w2 p95        {_percentile([r.mse_w2 for r in measured], 95):.2e}\n"
    )


def write_json_report(path, rounds: list[RoundMetrics], cfg: FFNConfig, *,
                      warmup: int, k: int = SLALOM_K,
                      link_gbps: Optional[float] = None) -> None:
    measured = rounds[warmup:] if warmup > 0 else rounds
    rates = [compute_round_rates(r, cfg, k) for r in measured]

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    mean_e2e_ms = _mean([r.end_to_end_t for r in measured]) or 1e-9
    wire_payload_bytes = _mean([r.bytes_recv for r in measured])
    wire_only_ms = _mean(
        [max(0.0, r.wire_recv_t - r.gpu_forward_t) for r in measured])
    wire_eff_gbps = (wire_payload_bytes * 8.0 / (wire_only_ms / 1000.0) / 1e9) \
        if wire_only_ms > 0 else 0.0
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
        # Shapes/dtypes/sizes of the tensors the worker streams back each round
        # (constant across rounds; sampled from the first measured round).
        "wire_tensors": next((r.recv_tensors for r in measured if r.recv_tensors), {}),
        # Ideal-vs-measured transfer estimate for the per-round payload.
        "wire_payload_bytes": wire_payload_bytes,
        "mean_wire_only_ms": wire_only_ms,
        "mean_wire_effective_gbps": wire_eff_gbps,
        "link_gbps": link_gbps,
        "wire_efficiency": (wire_eff_gbps / link_gbps) if link_gbps else None,
    }
    payload = {
        "config": asdict(cfg),
        "per_round": [asdict(r) for r in rounds],
        "summary": summary,
    }
    import pathlib as _p
    _p.Path(path).write_text(json.dumps(payload, indent=2))


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
                 threshold: float = 1e-3, k: int = SLALOM_K,
                 pipeline: bool = False):
        self.host = host
        self.port = port
        self.config = config
        self.threshold = threshold
        self.k = k
        self.pipeline = pipeline
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

    @staticmethod
    def _timed(fn, *args):
        t0 = time.perf_counter()
        mse = fn(*args)
        return mse, (time.perf_counter() - t0) * 1000.0

    def _expected_shape(self, op_tag: int) -> tuple:
        out = self.config.inter if op_tag in (OP_W1, OP_W3) else self.config.hidden
        return (self.config.batch, self.config.seq, out)

    def run_round(self, request_id: int, input_seed: int) -> RoundMetrics:
        assert self.sock is not None, "call connect_and_load() first"
        rm = RoundMetrics(request_id=request_id)
        rm.bytes_recv_predicted = self.predicted_recv_bytes()

        x_cpu = self.reproduce_input_cpu(input_seed)

        t_start = time.perf_counter()
        req_body = pack_forward_req(request_id, input_seed,
                                     self.config.batch, self.config.seq)
        rm.bytes_sent = send_msg(self.sock, MSG_FORWARD_REQ, req_body)

        # Receive 3 ACTIVATION + 1 FORWARD_DONE. In pipeline mode, each
        # SLALOM check is submitted to the pool the moment its activation
        # arrives — so verification of y1 overlaps with the receive of y3,
        # and verification of y3 overlaps with the receive of y2.
        t_wire_start = time.perf_counter()
        bytes_recv = 0
        first_act_t: Optional[float] = None
        first_submit_t: Optional[float] = None
        acts: dict[int, torch.Tensor] = {}
        futures: dict[int, object] = {}
        gpu_forward_t_ms = 0.0
        done = False
        while not done or len(acts) < 3:
            mt, body = recv_msg(self.sock)
            bytes_recv += 8 + len(body)
            if mt == MSG_ACTIVATION:
                if first_act_t is None:
                    first_act_t = time.perf_counter()
                d = unpack_activation(body)
                op_tag = d["op_tag"]
                tensor = d["tensor"]
                if tuple(tensor.shape) != self._expected_shape(op_tag):
                    raise WireProtocolError(
                        f"op_tag={op_tag}: expected shape "
                        f"{self._expected_shape(op_tag)}, got {tuple(tensor.shape)}"
                    )
                rm.recv_tensors[op_tag] = {
                    "shape": list(tensor.shape),
                    "dtype": _DTYPE_NAME[d["dtype_id"]],
                    "bytes": 8 + len(body),  # framed message size (header + body)
                }
                t32 = tensor.to(torch.float32)
                acts[op_tag] = t32
                if self.pipeline:
                    if first_submit_t is None:
                        first_submit_t = time.perf_counter()
                    if op_tag == OP_W1:
                        futures[OP_W1] = self.pool.submit(
                            self._timed, slalom_verify_safe,
                            x_cpu, t32, self.s_w1, self.s_tilde_w1)
                    elif op_tag == OP_W3:
                        futures[OP_W3] = self.pool.submit(
                            self._timed, slalom_verify_safe,
                            x_cpu, t32, self.s_w3, self.s_tilde_w3)
                    elif op_tag == OP_W2:
                        # y1 and y3 already arrived (wire order is W1, W3, W2).
                        gated_cpu = F.silu(acts[OP_W1]) * acts[OP_W3]
                        futures[OP_W2] = self.pool.submit(
                            self._timed, slalom_verify_safe,
                            gated_cpu, t32, self.s_w2, self.s_tilde_w2)
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

        if self.pipeline:
            # Verifications were kicked off during the receive loop. Collect
            # the (still-running or finished) futures. cpu_verify_t spans from
            # the first submit to the last result — it overlaps wire_recv_t,
            # which is why sum_pct comes out > 100% in this mode.
            rm.mse_w1, rm.cpu_verify_w1_t = futures[OP_W1].result()
            rm.mse_w3, rm.cpu_verify_w3_t = futures[OP_W3].result()
            rm.mse_w2, rm.cpu_verify_w2_t = futures[OP_W2].result()
            t_verify_end = time.perf_counter()
            rm.cpu_verify_t = (t_verify_end - (first_submit_t or t_wire_end)) * 1000.0
        else:
            # Sequential: verify only after the full receive completes.
            t_verify_start = time.perf_counter()
            y1, y3, y2 = acts[OP_W1], acts[OP_W3], acts[OP_W2]
            f_w1 = self.pool.submit(self._timed, slalom_verify_safe,
                                    x_cpu, y1, self.s_w1, self.s_tilde_w1)
            f_w3 = self.pool.submit(self._timed, slalom_verify_safe,
                                    x_cpu, y3, self.s_w3, self.s_tilde_w3)
            rm.mse_w1, rm.cpu_verify_w1_t = f_w1.result()
            rm.mse_w3, rm.cpu_verify_w3_t = f_w3.result()
            gated_cpu = F.silu(y1) * y3
            f_w2 = self.pool.submit(self._timed, slalom_verify_safe,
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


def default_threshold(wire_dtype_id: int, inter: int) -> float:
    """Pick an MSE threshold above the wire's numeric noise floor.

    fp32 wire round-trips exactly, so its noise floor is ~1e-15 — 1e-3 is
    plenty. fp16 wire quantizes each activation; that error, summed over
    `inter` (and `hidden`) in the SLALOM projection, grows roughly linearly
    with the dims, landing around 2-4e-3 at inter=11008. Scale with `inter`
    so larger models still pass clean while forged outputs (orders of
    magnitude above the floor) still fail.
    """
    if wire_dtype_id == DTYPE_FP32:
        return 1e-3
    return max(1e-3, inter * 2e-6)  # fp16


def run_coordinator(host: str, port: int, args) -> int:
    cfg = FFNConfig(
        hidden=args.hidden, inter=args.inter,
        batch=args.batch, seq=args.seq,
        wire_dtype=_NAME_TO_DTYPE[args.wire_dtype],
        weight_seed=args.weight_seed,
    )
    threshold = args.threshold
    if threshold is None:
        threshold = default_threshold(cfg.wire_dtype, cfg.inter)
    coord = Coordinator(host=host, port=port, config=cfg,
                         threshold=threshold, k=SLALOM_K,
                         pipeline=args.pipeline)
    try:
        coord.connect_and_load()
        rounds = coord.run_many(rounds=args.rounds)
    finally:
        coord.close()
    print(format_summary(rounds, cfg, warmup=args.warmup, k=SLALOM_K,
                         pipelined=args.pipeline, link_gbps=args.link_gbps))
    if args.json_report:
        write_json_report(args.json_report, rounds, cfg,
                          warmup=args.warmup, k=SLALOM_K,
                          link_gbps=args.link_gbps)
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
    if args.pipeline:
        worker_cmd.append("--pipeline")
    proc = subprocess.Popen(worker_cmd, stderr=subprocess.PIPE)
    try:
        _wait_port(port)
        return run_coordinator("127.0.0.1", port, args)
    finally:
        # Coordinator.close() already sent CLOSE; give worker a moment to
        # exit cleanly before we SIGTERM it. Otherwise every clean run prints
        # a spurious "worker exited with -15" line.
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Only surface stderr for genuinely unexpected non-zero exits.
        # -SIGTERM (-15) means we deliberately killed it after our own timeout.
        rc = proc.returncode
        if rc is not None and rc != 0 and rc != -15:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            sys.stderr.write(f"worker exited with {rc}\n{err}\n")


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
    p.add_argument("--threshold", type=float, default=None,
                   help="SLALOM MSE threshold; default scales with wire dtype "
                        "and dims (1e-3 for fp32, ~inter*2e-6 for fp16)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject-fault", default="none",
                   choices=["none", "flip_y1", "scale_y2", "drop_silu"])
    p.add_argument("--json-report", default=None)
    p.add_argument("--link-gbps", type=float, default=None,
                   help="nominal link bandwidth (Gbit/s); adds an ideal-vs-"
                        "measured transfer comparison and a link-efficiency %% "
                        "to the summary")
    p.add_argument("--pipeline", action="store_true",
                   help="overlap worker send with compute and coordinator "
                        "verify with receive (sum_pct then exceeds 100%%)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the worker's per-round progress log")
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
