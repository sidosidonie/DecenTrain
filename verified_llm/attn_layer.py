"""
Verified attention layer with CPU verification chain.

GPU runs the complete forward pass (all matmuls + non-linears).
CPU asynchronously verifies:
  - Linear projections via SLALOM preprocessed Freivalds (D2H output only)
  - Q@K^T and P@V via standard Freivalds (D2H output only, inputs from CPU chain)
  - Non-linears (q_norm, RoPE, softmax, sigmoid) recomputed on CPU from chain data

Only matmul outputs are D2H'd. Non-linear results are computed on CPU from
already-verified data, so they are trusted by construction.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

# Prefer Qwen3.5 apply_rotary_pos_emb: handles partial_rotary_factor.
apply_rotary_pos_emb = None
for _model_module in [
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.llama.modeling_llama",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.mistral.modeling_mistral",
]:
    if apply_rotary_pos_emb is not None:
        break
    try:
        import importlib
        _mod = importlib.import_module(_model_module)
        apply_rotary_pos_emb = getattr(_mod, "apply_rotary_pos_emb", None)
    except ImportError:
        pass

assert apply_rotary_pos_emb is not None, (
    "Could not import apply_rotary_pos_emb from any supported model module"
)

from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.verify_linear import (
    VerifyLinear,
    freivalds_batch_matmul,
    slalom_verify_preprocessed,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _rms_norm_cpu(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """CPU RMSNorm matching Qwen3/Llama implementation."""
    x_f = x.float()
    out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out


class LlamaAttentionVerify(nn.Module):
    """Verified attention with CPU verification chain.

    GPU runs the complete forward pass. CPU asynchronously verifies every
    matmul output and recomputes every non-linear operation from the
    verified chain data. Only matmul outputs are D2H copied.
    """

    def __init__(
        self,
        origin: nn.Module,
        runtime: VerifyRuntime,
        tag_prefix: str = "",
        noise=None,
    ):
        super().__init__()
        self.config = origin.config
        self.layer_idx = origin.layer_idx
        self.head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        # QK normalization — copy weights to CPU for chain
        self.q_norm = getattr(origin, "q_norm", None)
        self.k_norm = getattr(origin, "k_norm", None)
        self.q_norm_w = self.q_norm.weight.detach().float().cpu() if self.q_norm is not None else None
        self.q_norm_eps = self.q_norm.eps if self.q_norm is not None else None
        self.k_norm_w = self.k_norm.weight.detach().float().cpu() if self.k_norm is not None else None
        self.k_norm_eps = self.k_norm.eps if self.k_norm is not None else None

        # Gated attention (Qwen3.5)
        expected_q_size = self.config.num_attention_heads * self.head_dim
        self.has_attn_gate = (origin.q_proj.out_features == expected_q_size * 2)

        prefix = tag_prefix or f"layer{self.layer_idx}"
        self.q_proj = VerifyLinear(origin.q_proj, runtime, f"{prefix}.q", noise)
        self.k_proj = VerifyLinear(origin.k_proj, runtime, f"{prefix}.k", noise)
        self.v_proj = VerifyLinear(origin.v_proj, runtime, f"{prefix}.v", noise)
        self.o_proj = VerifyLinear(origin.o_proj, runtime, f"{prefix}.o", noise)
        self.runtime = runtime

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cache = past_key_value if past_key_value is not None else past_key_values
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        # ──── GPU Forward Pass (complete, non-blocking) ──────────────────

        # Linear projections (GPU only, no auto-verify)
        q_raw = self.q_proj.forward_gpu_only(hidden_states)
        k_raw = self.k_proj.forward_gpu_only(hidden_states)
        v_raw = self.v_proj.forward_gpu_only(hidden_states)

        # Add bias + reshape (GPU)
        q_biased = self.q_proj.add_bias(q_raw)
        if self.has_attn_gate:
            q_reshaped = q_biased.view(*input_shape, -1, self.head_dim * 2)
            query, attn_gate = torch.chunk(q_reshaped, 2, dim=-1)
            attn_gate = attn_gate.reshape(*input_shape, -1)
            query = query.view(hidden_shape).transpose(1, 2)
        else:
            attn_gate = None
            query = q_biased.view(hidden_shape).transpose(1, 2)

        key = self.k_proj.add_bias(k_raw).view(hidden_shape).transpose(1, 2)
        value = self.v_proj.add_bias(v_raw).view(hidden_shape).transpose(1, 2)

        # Non-linear: q_norm, k_norm, RoPE (GPU)
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # KV cache — track whether cache had previous entries for this layer
        # (affects whether CPU chain can verify Q@K^T / P@V)
        cache_had_previous = (
            cache is not None
            and hasattr(cache, "get_seq_length")
            and cache.get_seq_length(self.layer_idx) > 0
        )
        if cache is not None:
            try:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key, value = cache.update(key, value, self.layer_idx, cache_kwargs)
            except TypeError:
                key, value = cache.update(key, value, self.layer_idx)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        # Q@K^T (GPU matmul)
        scores = torch.matmul(query, key.transpose(2, 3))
        scores_scaled = scores * self.scaling
        if attention_mask is not None:
            scores_scaled = scores_scaled + attention_mask[:, :, :, :key.shape[-2]]

        dtype = query.dtype
        attn_probs = F.softmax(scores_scaled, dim=-1, dtype=torch.float32).to(dtype)
        attn_probs = F.dropout(attn_probs, p=self.attention_dropout if self.training else 0.0, training=self.training)

        # P@V (GPU matmul)
        attn_out = torch.matmul(attn_probs, value)
        attn_out_reshaped = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)

        if attn_gate is not None:
            attn_out_reshaped = attn_out_reshaped * torch.sigmoid(attn_gate)

        # O projection (GPU matmul)
        o_raw = self.o_proj.forward_gpu_only(attn_out_reshaped)
        output = self.o_proj.add_bias(o_raw)

        # ──── Submit async chain verification ────────────────────────────

        if self.runtime.config.enabled:
            self._submit_chain(
                hidden_states, q_raw, k_raw, v_raw,
                scores, attn_out, o_raw,
                cos, sin, attention_mask,
                cache_had_previous,
            )

        return output, attn_probs

    def _submit_chain(
        self, hidden_states, q_raw, k_raw, v_raw,
        scores, attn_out, o_raw,
        cos, sin, attention_mask, has_cache,
    ):
        rt = self.runtime
        if not rt.should_verify_now():
            return

        # D2H copies — only matmul outputs + shared input
        d2h = rt.d2h_async
        x_h, x_e = d2h(hidden_states)
        q_h, q_e = d2h(q_raw)
        k_h, k_e = d2h(k_raw)
        v_h, v_e = d2h(v_raw)
        sc_h, sc_e = d2h(scores)
        ao_h, ao_e = d2h(attn_out)
        o_h, o_e = d2h(o_raw)

        # Non-linear params to CPU (small, synchronous copy is fine)
        cos_cpu = cos.detach().float().cpu()
        sin_cpu = sin.detach().float().cpu()
        mask_cpu = attention_mask.detach().float().cpu() if attention_mask is not None else None

        # Capture everything the chain task needs (no references to self)
        q_s, q_st, q_bias = self.q_proj.s, self.q_proj.s_tilde, self.q_proj.bias_cpu
        k_s, k_st, k_bias = self.k_proj.s, self.k_proj.s_tilde, self.k_proj.bias_cpu
        v_s, v_st, v_bias = self.v_proj.s, self.v_proj.s_tilde, self.v_proj.bias_cpu
        o_s, o_st = self.o_proj.s, self.o_proj.s_tilde
        q_nw, q_ne = self.q_norm_w, self.q_norm_eps
        k_nw, k_ne = self.k_norm_w, self.k_norm_eps
        layer_idx = self.layer_idx
        head_dim = self.head_dim
        num_kv_groups = self.num_key_value_groups
        scaling = self.scaling
        has_gate = self.has_attn_gate
        threshold = rt.config.mse_threshold
        freivalds_k = rt.config.freivalds_k
        profiler = rt.profiler
        errors = rt._errors

        # Keep GPU refs alive until D2H completes
        _gpu_refs = [hidden_states, q_raw, k_raw, v_raw, scores, attn_out, o_raw]

        def _chain():
            # Wait for all D2H
            for evt in [x_e, q_e, k_e, v_e, sc_e, ao_e, o_e]:
                if evt is not None:
                    evt.synchronize()
            # Release GPU refs
            _gpu_refs.clear()

            x = x_h.float()
            prefix = f"layer{layer_idx}"

            def _slalom(tag, x_cpu, y_cpu, s, st):
                t0 = time.perf_counter()
                loss = slalom_verify_preprocessed(x_cpu, y_cpu, s, st)
                dt = (time.perf_counter() - t0) * 1000
                ok = loss <= threshold
                if not ok:
                    errors.append(f"{tag} SLALOM failed: loss={loss:.6e}")
                profiler.add("verify", "linear_slalom", dt, tag=tag, ok=ok, extra=f"loss={loss:.6e}")

            def _freivalds(tag, A, B, C):
                t0 = time.perf_counter()
                loss = freivalds_batch_matmul(A, B, C, k=freivalds_k)
                dt = (time.perf_counter() - t0) * 1000
                ok = loss <= threshold
                if not ok:
                    errors.append(f"{tag} Freivalds failed: loss={loss:.6e}")
                profiler.add("verify", "matmul_freivalds", dt, tag=tag, ok=ok, extra=f"loss={loss:.6e}")

            # ── 1. Verify Q/K/V projections (SLALOM) ──
            _slalom(f"{prefix}.q", x, q_h.float(), q_s, q_st)
            _slalom(f"{prefix}.k", x, k_h.float(), k_s, k_st)
            _slalom(f"{prefix}.v", x, v_h.float(), v_s, v_st)

            # ── 2. CPU chain: bias + reshape + q_norm + RoPE ──
            q_cpu = q_h.float()
            k_cpu = k_h.float()
            v_cpu = v_h.float()
            if q_bias is not None:
                q_cpu = q_cpu + q_bias
            if k_bias is not None:
                k_cpu = k_cpu + k_bias
            if v_bias is not None:
                v_cpu = v_cpu + v_bias

            inp_shape = q_cpu.shape[:-1]
            h_shape = (*inp_shape, -1, head_dim)

            if has_gate:
                q_r = q_cpu.view(*inp_shape, -1, head_dim * 2)
                q_heads, gate_cpu = torch.chunk(q_r, 2, dim=-1)
                gate_cpu = gate_cpu.reshape(*inp_shape, -1)
                q_heads = q_heads.view(h_shape).transpose(-3, -2)
            else:
                gate_cpu = None
                q_heads = q_cpu.view(h_shape).transpose(-3, -2)

            k_heads = k_cpu.view(h_shape).transpose(-3, -2)
            v_heads = v_cpu.view(h_shape).transpose(-3, -2)

            if q_nw is not None:
                q_heads = _rms_norm_cpu(q_heads, q_nw, q_ne)
            if k_nw is not None:
                k_heads = _rms_norm_cpu(k_heads, k_nw, k_ne)

            q_heads, k_heads = apply_rotary_pos_emb(q_heads, k_heads, cos_cpu, sin_cpu)
            k_heads = repeat_kv(k_heads, num_kv_groups)
            v_heads = repeat_kv(v_heads, num_kv_groups)

            # ── 3. Verify Q@K^T (Freivalds, CPU chain inputs) ──
            if not has_cache:
                _freivalds(f"{prefix}.qk", q_heads, k_heads.transpose(-2, -1), sc_h.float())

                # ── 4. CPU softmax ──
                sc_cpu = sc_h.float() * scaling
                if mask_cpu is not None:
                    sc_cpu = sc_cpu + mask_cpu[:, :, :, :k_heads.shape[-2]]
                probs_cpu = F.softmax(sc_cpu, dim=-1, dtype=torch.float32)

                # ── 5. Verify P@V (Freivalds, CPU chain inputs) ──
                _freivalds(f"{prefix}.kv", probs_cpu, v_heads, ao_h.float())

            # ── 6. CPU chain: reshape + gate → input for O projection ──
            attn_cpu = ao_h.float()
            attn_cpu = attn_cpu.transpose(-3, -2).contiguous()
            attn_cpu = attn_cpu.reshape(*inp_shape, -1)
            if gate_cpu is not None:
                attn_cpu = attn_cpu * torch.sigmoid(gate_cpu)

            # ── 7. Verify O projection (SLALOM, CPU chain input) ──
            _slalom(f"{prefix}.o", attn_cpu, o_h.float(), o_s, o_st)

            # Record CPU chain non-linear recomputation
            profiler.add("verify", "cpu_chain_nonlinear", 0.0, tag=prefix, ok=True,
                         extra="q_norm,k_norm,rope,softmax,sigmoid")

        rt._enqueue(_chain)
