"""
Verified MLP layer with CPU verification chain.

GPU runs the complete MLP forward (gate/up/down projections + SiLU + elementwise).
CPU asynchronously verifies — only matmul OUTPUTS are D2H copied.
The MLP input comes from the CPU chain (post-attention state + layernorm).

Data flow:
  cpu_state['post_attn_{idx}'] ──► post_attention_layernorm (CPU)
    ──► verify gate/up outputs (D2H) via SLALOM
    ──► silu + elementwise multiply (CPU)
    ──► verify down output (D2H) via SLALOM
    ──► residual (CPU)
    ──► cpu_state['layer_input_{idx+1}']
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verified_core.runtime import VerifyRuntime
from verified_core.verify_linear import VerifyLinear, slalom_verify_preprocessed


def _rms_norm_cpu(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """CPU RMSNorm matching Qwen3/Llama implementation."""
    x_f = x.float()
    out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out


class LlamaMLPVerify(nn.Module):
    def __init__(
        self,
        origin: nn.Module,
        runtime: VerifyRuntime,
        tag_prefix: str = "mlp",
        noise_scale=None,
        *,
        layer_idx: int = 0,
        post_attn_ln_weight: Optional[torch.Tensor] = None,
        post_attn_ln_eps: float = 1e-6,
    ):
        super().__init__()
        self.gate_proj = VerifyLinear(origin.gate_proj, runtime, f"{tag_prefix}.gate", noise_scale)
        self.up_proj = VerifyLinear(origin.up_proj, runtime, f"{tag_prefix}.up", noise_scale)
        self.down_proj = VerifyLinear(origin.down_proj, runtime, f"{tag_prefix}.down", noise_scale)
        self.runtime = runtime
        self.tag_prefix = tag_prefix

        # CPU chain: layer index and post-attention layernorm weights
        self._chain_layer_idx = layer_idx
        self._post_attn_ln_weight = post_attn_ln_weight   # float32, CPU
        self._post_attn_ln_eps = post_attn_ln_eps

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
            self._submit_chain(gate_raw, up_raw, down_raw)

        return down

    # ------------------------------------------------------------------
    # CPU verification chain — only matmul OUTPUTS are D2H copied
    # ------------------------------------------------------------------

    def _submit_chain(self, gate_raw, up_raw, down_raw):
        rt = self.runtime

        # D2H copies — ONLY matmul outputs (no input D2H!)
        d2h = rt.d2h_async
        g_h, g_e = d2h(gate_raw)
        u_h, u_e = d2h(up_raw)
        d_h, d_e = d2h(down_raw)

        # Capture for closure
        g_s, g_st, g_bias = self.gate_proj.s, self.gate_proj.s_tilde, self.gate_proj.bias_cpu
        u_s, u_st, u_bias = self.up_proj.s, self.up_proj.s_tilde, self.up_proj.bias_cpu
        d_s, d_st, d_bias = self.down_proj.s, self.down_proj.s_tilde, self.down_proj.bias_cpu
        prefix = self.tag_prefix
        layer_idx = self._chain_layer_idx
        post_attn_ln_w = self._post_attn_ln_weight
        post_attn_ln_eps = self._post_attn_ln_eps
        threshold = rt.config.mse_threshold
        profiler = rt.profiler
        errors = rt._errors

        _gpu_refs = [gate_raw, up_raw, down_raw]

        def _chain():
            for evt in [g_e, u_e, d_e]:
                if evt is not None:
                    evt.synchronize()
            _gpu_refs.clear()

            # ── Get post-attention state from CPU chain (NOT from GPU!) ──
            post_attn = rt.cpu_state_get(f"post_attn_{layer_idx}")
            post_attn_f = post_attn.float()

            # ── Post-attention layernorm on CPU ──
            x_cpu = _rms_norm_cpu(post_attn_f, post_attn_ln_w, post_attn_ln_eps)

            def _slalom(tag, x_in, y_out, s, st):
                t0 = time.perf_counter()
                loss = slalom_verify_preprocessed(x_in, y_out, s, st)
                dt = (time.perf_counter() - t0) * 1000
                ok = loss <= threshold
                if not ok:
                    errors.append(f"{tag} SLALOM failed: loss={loss:.6e}")
                profiler.add("verify", "linear_slalom", dt, tag=tag, ok=ok, extra=f"loss={loss:.6e}")

            # ── 1. Verify gate and up projections (SLALOM) — input from CPU chain ──
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

            # ── 4. Compute MLP output + residual on CPU ──
            mlp_output = d_h.float()
            if d_bias is not None:
                mlp_output = mlp_output + d_bias
            layer_output = post_attn_f + mlp_output          # residual connection

            # Store for next layer's attention chain
            rt.cpu_state_set(f"layer_input_{layer_idx + 1}", layer_output)

            profiler.add("verify", "cpu_chain_nonlinear", 0.0, tag=prefix, ok=True,
                         extra="layernorm,silu,mul,residual")

        rt._enqueue(_chain)
