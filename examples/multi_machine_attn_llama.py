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
