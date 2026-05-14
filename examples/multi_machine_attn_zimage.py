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
