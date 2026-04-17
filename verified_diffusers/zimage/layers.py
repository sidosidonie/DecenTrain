"""
Verified linear and matmul wrappers for diffusion transformer blocks.

VerifyLinearModule: SLALOM preprocessed Freivalds for fixed-weight linear layers.
VerifyMatmul: Standard Freivalds for dynamic matmuls (Q@K^T, P@V).

Both support two modes:
  - forward(): GPU compute + auto-submit verification (standalone use)
  - forward_gpu_only(): GPU compute only; caller handles verification via CPU chain
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verified_core.runtime import VerifyRuntime
from verified_core.verify_linear import slalom_precompute, slalom_verify_preprocessed


class VerifyLinearModule(nn.Module):
    """Verified nn.Linear with SLALOM preprocessed Freivalds."""

    def __init__(self, linear: nn.Linear, runtime: VerifyRuntime, tag: str):
        super().__init__()
        self.linear = linear
        self.runtime = runtime
        self.tag = tag

        # SLALOM preprocessing — round-trip through GPU dtype to match rounding
        cfg = runtime.config
        self.s, self.s_tilde = slalom_precompute(
            linear.weight.detach().t().float().to("cpu"),
            k=cfg.freivalds_k,
            s_range=cfg.slalom_s_range,
            gpu_dtype=linear.weight.dtype,
        )
        self.bias_cpu = linear.bias.detach().float().to("cpu") if linear.bias is not None else None

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GPU forward + auto-submit SLALOM verification."""
        y_no_bias = self.runtime.profile_cuda_compute(
            tag=self.tag, op_name="linear", shape=str(tuple(x.shape)),
            fn=lambda: F.linear(x, self.linear.weight, bias=None),
        )
        self.runtime.submit_linear_preprocessed(
            tag=self.tag, x_gpu=x, y_gpu=y_no_bias,
            s=self.s, s_tilde=self.s_tilde,
        )
        if self.linear.bias is None:
            return y_no_bias
        return y_no_bias + self.linear.bias

    def forward_gpu_only(self, x: torch.Tensor) -> torch.Tensor:
        """GPU forward only, no verification. Caller handles via CPU chain."""
        y_no_bias = F.linear(x, self.linear.weight, bias=None)
        if self.linear.bias is None:
            return y_no_bias
        return y_no_bias + self.linear.bias

    def add_bias_cpu(self, y_cpu: torch.Tensor) -> torch.Tensor:
        """Add bias on CPU for chain verification."""
        if self.bias_cpu is not None:
            return y_cpu + self.bias_cpu
        return y_cpu


class VerifyMatmul(nn.Module):
    """Verified matmul for dynamic operands (e.g. Q@K^T, P@V)."""

    def __init__(self, runtime: VerifyRuntime, tag: str):
        super().__init__()
        self.runtime = runtime
        self.tag = tag

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """GPU matmul + auto-submit Freivalds verification."""
        out = self.runtime.profile_cuda_compute(
            tag=self.tag, op_name="matmul",
            shape=f"{tuple(a.shape)}x{tuple(b.shape)}",
            fn=lambda: torch.matmul(a, b),
        )
        self.runtime.submit_matmul(self.tag, a, b, out)
        return out

    def forward_gpu_only(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """GPU matmul only, no verification. Caller handles via CPU chain."""
        return torch.matmul(a, b)
