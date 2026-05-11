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


def main() -> int:  # pragma: no cover - filled in last task
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
