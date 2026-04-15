"""
Verified Z-Image feedforward (SwiGLU MLP) with CPU verification chain.

GPU runs complete forward: w1(x), w3(x), silu(w1) * w3, w2(gated).
CPU asynchronously verifies all 3 linear projections via SLALOM and
recomputes silu + elementwise multiply from verified chain data.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from verified_diffusers.zimage.layers import VerifyLinearModule
from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.verify_linear import slalom_verify_preprocessed


class VerifiedZImageFeedForward(nn.Module):
    def __init__(self, feed_forward: nn.Module, runtime: VerifyRuntime, tag_prefix: str):
        super().__init__()
        self.runtime = runtime
        self.w1 = VerifyLinearModule(feed_forward.w1, runtime, f"{tag_prefix}.w1")
        self.w2 = VerifyLinearModule(feed_forward.w2, runtime, f"{tag_prefix}.w2")
        self.w3 = VerifyLinearModule(feed_forward.w3, runtime, f"{tag_prefix}.w3")
        self.tag_prefix = tag_prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ──── GPU Forward (complete, non-blocking) ───────────────────

        w1_raw = self.w1.forward_gpu_only(x)
        w3_raw = self.w3.forward_gpu_only(x)
        gated = F.silu(w1_raw) * w3_raw
        w2_raw = self.w2.forward_gpu_only(gated)

        # ──── Submit async chain verification ────────────────────────

        if self.runtime.config.enabled:
            self._submit_chain(x, w1_raw, w3_raw, w2_raw)

        return w2_raw

    def _submit_chain(self, x, w1_raw, w3_raw, w2_raw):
        rt = self.runtime
        if not rt.should_verify_now():
            return

        d2h = rt.d2h_async
        x_h, x_e = d2h(x)
        w1_h, w1_e = d2h(w1_raw)
        w3_h, w3_e = d2h(w3_raw)
        w2_h, w2_e = d2h(w2_raw)

        s1, st1 = self.w1.s, self.w1.s_tilde
        s3, st3 = self.w3.s, self.w3.s_tilde
        s2, st2 = self.w2.s, self.w2.s_tilde
        prefix = self.tag_prefix
        threshold = rt.config.mse_threshold
        profiler = rt.profiler
        errors = rt._errors

        _gpu_refs = [x, w1_raw, w3_raw, w2_raw]

        def _chain():
            for evt in [x_e, w1_e, w3_e, w2_e]:
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

            # 1. Verify w1 and w3 projections (shared input)
            _slalom(f"{prefix}.w1", x_cpu, w1_h.float(), s1, st1)
            _slalom(f"{prefix}.w3", x_cpu, w3_h.float(), s3, st3)

            # 2. CPU chain: silu(w1) * w3
            gated_cpu = F.silu(w1_h.float()) * w3_h.float()

            # 3. Verify w2 projection (CPU chain input)
            _slalom(f"{prefix}.w2", gated_cpu, w2_h.float(), s2, st2)

            profiler.add("verify", "cpu_chain_nonlinear", 0.0, tag=prefix, ok=True, extra="silu,mul")

        rt._enqueue(_chain)
