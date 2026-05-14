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
            outs[OP_O] = outs[OP_O] * 2.0
        elif self.inject_fault == "scale_w2":
            outs[OP_W2] = outs[OP_W2] * 2.0
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
