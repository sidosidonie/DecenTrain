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
            return q, k, v, o * 2.0
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
