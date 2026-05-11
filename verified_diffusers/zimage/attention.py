"""
Verified Z-Image attention with CPU verification chain.

GPU runs the complete attention forward. CPU asynchronously verifies:
  - Q/K/V/O projections via SLALOM (D2H output only)
  - Q@K^T and P@V via Freivalds (D2H output only, inputs from CPU chain)
  - Non-linears (norm_q/k, RoPE, softmax) recomputed on CPU

The chain input is read from runtime.cpu_state[input_state_key]. If the
caller does not provide a key (single-process tests), forward() seeds
the default key via D2H of hidden_states. The multi-machine coordinator
always passes an explicit key populated from CPU recompute, so the
attention input is never shipped over the wire.
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from verified_diffusers.zimage.layers import VerifyLinearModule
from verified_core.runtime import VerifyRuntime
from verified_core.verify_linear import freivalds_batch_matmul, slalom_verify_preprocessed


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


def _apply_rotary_emb_cpu(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """CPU version of apply_rotary_emb."""
    x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
    fc = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x * fc).flatten(3)
    return x_out


class VerifiedZImageAttention(nn.Module):
    def __init__(self, attention: Attention, runtime: VerifyRuntime, tag_prefix: str):
        super().__init__()
        self.runtime = runtime
        self.tag_prefix = tag_prefix
        self.heads = attention.heads
        self.norm_q = attention.norm_q
        self.norm_k = attention.norm_k

        # Copy norm weights/eps to CPU for chain
        self.norm_q_w = self.norm_q.weight.detach().float().cpu() if (self.norm_q is not None and self.norm_q.weight is not None) else None
        self.norm_q_eps = self.norm_q.eps if self.norm_q is not None else None
        self.norm_k_w = self.norm_k.weight.detach().float().cpu() if (self.norm_k is not None and self.norm_k.weight is not None) else None
        self.norm_k_eps = self.norm_k.eps if self.norm_k is not None else None

        self.to_q = VerifyLinearModule(attention.to_q, runtime, f"{tag_prefix}.to_q")
        self.to_k = VerifyLinearModule(attention.to_k, runtime, f"{tag_prefix}.to_k")
        self.to_v = VerifyLinearModule(attention.to_v, runtime, f"{tag_prefix}.to_v")
        self.to_out_linear = VerifyLinearModule(attention.to_out[0], runtime, f"{tag_prefix}.to_out")
        self.to_out_dropout = attention.to_out[1] if len(attention.to_out) > 1 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        *,
        input_state_key: Optional[str] = None,
    ) -> torch.Tensor:
        # ──── GPU Forward (complete, non-blocking) ───────────────────

        q_raw = self.to_q.forward_gpu_only(hidden_states)
        k_raw = self.to_k.forward_gpu_only(hidden_states)
        v_raw = self.to_v.forward_gpu_only(hidden_states)

        query = q_raw.unflatten(-1, (self.heads, -1))
        key = k_raw.unflatten(-1, (self.heads, -1))
        value = v_raw.unflatten(-1, (self.heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        q_t = query.permute(0, 2, 1, 3)
        k_t = key.permute(0, 2, 1, 3)
        v_t = value.permute(0, 2, 1, 3)

        head_dim = query.shape[-1]
        scores = torch.matmul(q_t, k_t.transpose(2, 3))
        scores_scaled = scores * (head_dim ** -0.5)
        if attention_mask is not None:
            scores_scaled = scores_scaled + attention_mask

        attn_probs = F.softmax(scores_scaled, dim=-1, dtype=torch.float32).to(dtype)

        attn_out = torch.matmul(attn_probs, v_t)
        attn_out = attn_out.permute(0, 2, 1, 3).flatten(2, 3).to(dtype)

        o_raw = self.to_out_linear.forward_gpu_only(attn_out)
        output = o_raw
        if self.to_out_dropout is not None:
            output = self.to_out_dropout(output)

        # ──── Submit async chain verification ────────────────────────

        if self.runtime.config.enabled:
            # Resolve the cpu_state key holding the chain input. If the caller
            # didn't pre-populate one, fall back to a D2H of hidden_states and
            # seed the default key ourselves — this keeps single-process tests
            # working while the multi-machine path always passes an explicit
            # key (and never triggers the input D2H).
            resolved_key = input_state_key or f"{self.tag_prefix}.input"
            if input_state_key is None:
                x_cpu, x_evt = self.runtime.d2h_async(hidden_states)
                self.runtime.cpu_state_set_d2h(resolved_key, x_cpu, x_evt)

            self._submit_chain(
                resolved_key, hidden_states, q_raw, k_raw, v_raw,
                scores, attn_out, o_raw,
                freqs_cis, attention_mask,
            )

        return output

    def _submit_chain(
        self, input_state_key, hidden_states, q_raw, k_raw, v_raw,
        scores, attn_out_flat, o_raw,
        freqs_cis, attention_mask,
    ):
        rt = self.runtime
        if not rt.should_verify_now():
            return

        d2h = rt.d2h_async
        q_h, q_e = d2h(q_raw)
        k_h, k_e = d2h(k_raw)
        v_h, v_e = d2h(v_raw)
        sc_h, sc_e = d2h(scores)
        ao_h, ao_e = d2h(attn_out_flat)
        o_h, o_e = d2h(o_raw)

        fc_cpu = freqs_cis.detach().cpu() if freqs_cis is not None else None
        mask_cpu = attention_mask.detach().float().cpu() if attention_mask is not None else None

        # Capture for closure
        tq_s, tq_st = self.to_q.s, self.to_q.s_tilde
        tk_s, tk_st = self.to_k.s, self.to_k.s_tilde
        tv_s, tv_st = self.to_v.s, self.to_v.s_tilde
        to_s, to_st = self.to_out_linear.s, self.to_out_linear.s_tilde
        nqw, nqe = self.norm_q_w, self.norm_q_eps
        nkw, nke = self.norm_k_w, self.norm_k_eps
        heads = self.heads
        prefix = self.tag_prefix
        threshold = rt.config.mse_threshold
        freivalds_k = rt.config.freivalds_k
        matmul_k = rt.config.matmul_freivalds_k or freivalds_k
        profiler = rt.profiler
        errors = rt._errors

        _gpu_refs = [hidden_states, q_raw, k_raw, v_raw, scores, attn_out_flat, o_raw]

        matmul_threshold = rt.config.matmul_mse_threshold or threshold

        # Split into two tasks: (1) linear SLALOM checks (fast), (2) matmul
        # Freivalds + non-linear chain (slow). This lets the thread pool
        # overlap SLALOM work from one layer with Freivalds from another.

        def _linear_checks():
            """Verify Q/K/V/O projections via SLALOM (fast, O(n*k))."""
            for evt in [q_e, k_e, v_e, ao_e, o_e]:
                if evt is not None:
                    evt.synchronize()

            x = rt.cpu_state_get(input_state_key).float()

            def _slalom(tag, x_in, y_out, s, st):
                t0 = time.perf_counter()
                loss = slalom_verify_preprocessed(x_in, y_out, s, st)
                dt = (time.perf_counter() - t0) * 1000
                ok = loss <= threshold
                if not ok:
                    errors.append(f"{tag} SLALOM failed: loss={loss:.6e}")
                profiler.add("verify", "linear_slalom", dt, tag=tag, ok=ok, extra=f"loss={loss:.6e}")

            _slalom(f"{prefix}.to_q", x, q_h.float(), tq_s, tq_st)
            _slalom(f"{prefix}.to_k", x, k_h.float(), tk_s, tk_st)
            _slalom(f"{prefix}.to_v", x, v_h.float(), tv_s, tv_st)
            _slalom(f"{prefix}.to_out", ao_h.float(), o_h.float(), to_s, to_st)

        def _matmul_chain():
            """Verify Q@K^T and P@V via Freivalds + non-linear recompute (slow)."""
            for evt in [q_e, k_e, v_e, sc_e, ao_e]:
                if evt is not None:
                    evt.synchronize()
            _gpu_refs.clear()

            def _freivalds(tag, A, B, C):
                t0 = time.perf_counter()
                loss = freivalds_batch_matmul(A, B, C, k=matmul_k)
                dt = (time.perf_counter() - t0) * 1000
                ok = loss <= matmul_threshold
                if not ok:
                    errors.append(f"{tag} Freivalds failed: loss={loss:.6e}")
                profiler.add("verify", "matmul_freivalds", dt, tag=tag, ok=ok, extra=f"loss={loss:.6e}")

            q_cpu = q_h.float().unflatten(-1, (heads, -1))
            k_cpu = k_h.float().unflatten(-1, (heads, -1))
            v_cpu = v_h.float().unflatten(-1, (heads, -1))

            if nqw is not None or nqe is not None:
                q_f = q_cpu.float()
                q_cpu = q_f * torch.rsqrt(q_f.pow(2).mean(-1, keepdim=True) + nqe)
                if nqw is not None:
                    q_cpu = q_cpu * nqw
            if nkw is not None or nke is not None:
                k_f = k_cpu.float()
                k_cpu = k_f * torch.rsqrt(k_f.pow(2).mean(-1, keepdim=True) + nke)
                if nkw is not None:
                    k_cpu = k_cpu * nkw

            if fc_cpu is not None:
                q_cpu = _apply_rotary_emb_cpu(q_cpu, fc_cpu)
                k_cpu = _apply_rotary_emb_cpu(k_cpu, fc_cpu)

            q_t = q_cpu.permute(0, 2, 1, 3)
            k_t = k_cpu.permute(0, 2, 1, 3)
            v_t = v_cpu.permute(0, 2, 1, 3)
            head_dim = q_cpu.shape[-1]

            _freivalds(f"{prefix}.qk", q_t, k_t.transpose(2, 3), sc_h.float())

            sc_cpu = sc_h.float() * (head_dim ** -0.5)
            if mask_cpu is not None:
                sc_cpu = sc_cpu + mask_cpu
            probs_cpu = F.softmax(sc_cpu, dim=-1, dtype=torch.float32)

            _freivalds(f"{prefix}.kv", probs_cpu, v_t, ao_h.float().unflatten(-1, (heads, -1)).permute(0, 2, 1, 3))

            profiler.add("verify", "cpu_chain_nonlinear", 0.0, tag=prefix, ok=True,
                         extra="norm_q,norm_k,rope,softmax")

        rt._enqueue(_linear_checks)
        rt._enqueue(_matmul_chain)
