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
            return q, k, v, o * 2.0
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
            # NO QK RMSNorm; scale output 10x so SLALOM detects at small test dims
            qh = apply_rotary_emb_zimage(qh, freqs.to(qh.device))
            kh = apply_rotary_emb_zimage(kh, freqs.to(kh.device))
            qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
            scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            attn_out = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
            return q, k, v, self.o_proj(attn_out) * 10.0
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
