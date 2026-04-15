"""
Verified MLP layer with CPU verification chain.

GPU runs the complete MLP forward (gate/up/down projections + SiLU + elementwise).
CPU asynchronously verifies:
  - gate/up/down projections via SLALOM (D2H output only)
  - SiLU and elementwise multiply recomputed on CPU from chain data

Only matmul outputs are D2H'd. Non-linear results (SiLU, gate*up) are computed
on CPU from already-verified data, so they are trusted by construction.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.verify_linear import VerifyLinear, slalom_verify_preprocessed


class LlamaMLPVerify(nn.Module):
    def __init__(
        self,
        origin: nn.Module,
        runtime: VerifyRuntime,
        tag_prefix: str = "mlp",
        noise_scale=None,
    ):
        super().__init__()
        self.gate_proj = VerifyLinear(origin.gate_proj, runtime, f"{tag_prefix}.gate", noise_scale)
        self.up_proj = VerifyLinear(origin.up_proj, runtime, f"{tag_prefix}.up", noise_scale)
        self.down_proj = VerifyLinear(origin.down_proj, runtime, f"{tag_prefix}.down", noise_scale)
        self.runtime = runtime
        self.tag_prefix = tag_prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ──── GPU Forward Pass (complete, non-blocking) ──────────────────

        gate_raw = self.gate_proj.forward_gpu_only(x)
        gate = self.gate_proj.add_bias(gate_raw)
        gate = F.silu(gate)

        up_raw = self.up_proj.forward_gpu_only(x)
        up = self.up_proj.add_bias(up_raw)

        down_input = gate * up

        down_raw = self.down_proj.forward_gpu_only(down_input)
        down = self.down_proj.add_bias(down_raw)

        # ──── Submit async chain verification ────────────────────────────

        if self.runtime.config.enabled:
            self._submit_chain(x, gate_raw, up_raw, down_raw)

        return down

    def _submit_chain(self, x, gate_raw, up_raw, down_raw):
        rt = self.runtime
        if not rt.should_verify_now():
            return

        # D2H copies — input (once) + 3 matmul outputs
        d2h = rt.d2h_async
        x_h, x_e = d2h(x)
        g_h, g_e = d2h(gate_raw)
        u_h, u_e = d2h(up_raw)
        d_h, d_e = d2h(down_raw)

        # Capture for closure
        g_s, g_st, g_bias = self.gate_proj.s, self.gate_proj.s_tilde, self.gate_proj.bias_cpu
        u_s, u_st, u_bias = self.up_proj.s, self.up_proj.s_tilde, self.up_proj.bias_cpu
        d_s, d_st = self.down_proj.s, self.down_proj.s_tilde
        prefix = self.tag_prefix
        threshold = rt.config.mse_threshold
        profiler = rt.profiler
        errors = rt._errors

        _gpu_refs = [x, gate_raw, up_raw, down_raw]

        def _chain():
            for evt in [x_e, g_e, u_e, d_e]:
                if evt is not None:
                    evt.synchronize()
            _gpu_refs.clear()

            x_cpu = x_h.float()

            def _slalom(tag, x_in, y_out, s, st):
                t0 = time.perf_counter()
                loss = slalom_verify_preprocessed(x_in, y_out, s, st)
                dt = (time.perf_counter() - t0) * 1000
                ok = loss <= threshold
                if not ok:
                    errors.append(f"{tag} SLALOM failed: loss={loss:.6e}")
                profiler.add("verify", "linear_slalom", dt, tag=tag, ok=ok, extra=f"loss={loss:.6e}")

            # ── 1. Verify gate and up projections (SLALOM, shared input) ──
            _slalom(f"{prefix}.gate", x_cpu, g_h.float(), g_s, g_st)
            _slalom(f"{prefix}.up", x_cpu, u_h.float(), u_s, u_st)

            # ── 2. CPU chain: silu(gate) * up ──
            gate_cpu = g_h.float()
            if g_bias is not None:
                gate_cpu = gate_cpu + g_bias
            gate_cpu = F.silu(gate_cpu)

            up_cpu = u_h.float()
            if u_bias is not None:
                up_cpu = up_cpu + u_bias

            down_input_cpu = gate_cpu * up_cpu

            # ── 3. Verify down projection (SLALOM, CPU chain input) ──
            _slalom(f"{prefix}.down", down_input_cpu, d_h.float(), d_s, d_st)

            profiler.add("verify", "cpu_chain_nonlinear", 0.0, tag=prefix, ok=True, extra="silu,mul")

        rt._enqueue(_chain)
