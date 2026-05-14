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
