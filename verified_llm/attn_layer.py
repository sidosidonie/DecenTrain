"""
Verified attention layer — replaces any Llama/Qwen/Mistral-compatible attention
module with manual attention + Freivalds verification via VerifyRuntime.

Supports models with:
  - q_proj, k_proj, v_proj, o_proj (Linear)
  - Optional q_norm, k_norm (e.g. Qwen3 QK normalization)
  - config with num_attention_heads, num_key_value_heads, head_dim
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

# Import apply_rotary_pos_emb from whichever model defines it.
# Qwen3.5 version is preferred: it handles partial_rotary_factor (splits
# q/k into rotary + passthrough parts), which is a superset of the Llama
# version that assumes full-dim rotary.
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

from verified_diffusers.zimage.layers import VerifyMatmul
from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.verify_linear import VerifyLinear


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttentionVerify(nn.Module):
    """Verified attention module compatible with Llama, Qwen2, Qwen3, Mistral, etc.

    Uses duck-typing: any module with q/k/v/o_proj Linear layers works.
    Optionally handles q_norm / k_norm (Qwen3-style QK normalization).
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

        # QK normalization (Qwen3 style) — copy if present
        self.q_norm = getattr(origin, "q_norm", None)
        self.k_norm = getattr(origin, "k_norm", None)

        # Qwen3.5-style attn output gating: q_proj outputs 2x (query + gate),
        # and attn_output *= sigmoid(gate) before o_proj.
        expected_q_size = self.config.num_attention_heads * self.head_dim
        self.has_attn_gate = (origin.q_proj.out_features == expected_q_size * 2)

        prefix = tag_prefix or f"layer{self.layer_idx}"
        self.q_proj = VerifyLinear(origin.q_proj, runtime, f"{prefix}.q", noise)
        self.k_proj = VerifyLinear(origin.k_proj, runtime, f"{prefix}.k", noise)
        self.v_proj = VerifyLinear(origin.v_proj, runtime, f"{prefix}.v", noise)
        self.o_proj = VerifyLinear(origin.o_proj, runtime, f"{prefix}.o", noise)
        self.qk_mm = VerifyMatmul(runtime, f"{prefix}.qk")
        self.kv_mm = VerifyMatmul(runtime, f"{prefix}.kv")
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
        # Support both Llama-style (past_key_value) and Qwen3.5-style (past_key_values)
        cache = past_key_value if past_key_value is not None else past_key_values

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # Projections — async verified via VerifyRuntime
        q_out = self.q_proj.forward(hidden_states)
        q_out = self.q_proj.add_bias(q_out)

        if self.has_attn_gate:
            # Qwen3.5: q_proj outputs [B, S, num_heads * head_dim * 2]
            # Split into query states and output gate.
            q_out = q_out.view(*input_shape, -1, self.head_dim * 2)
            query, attn_gate = torch.chunk(q_out, 2, dim=-1)
            attn_gate = attn_gate.reshape(*input_shape, -1)
            query = query.view(hidden_shape).transpose(1, 2)
        else:
            attn_gate = None
            query = q_out.view(hidden_shape).transpose(1, 2)

        key = self.k_proj.forward(hidden_states)
        key = self.k_proj.add_bias(key)
        key = key.view(hidden_shape).transpose(1, 2)

        value = self.v_proj.forward(hidden_states)
        value = self.v_proj.add_bias(value)
        value = value.view(hidden_shape).transpose(1, 2)

        # QK normalization (Qwen3) — applied before rotary embeddings
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        # Rotary embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # KV cache — Llama passes cache_kwargs, Qwen3.5 does not
        if cache is not None:
            try:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key, value = cache.update(key, value, self.layer_idx, cache_kwargs)
            except TypeError:
                key, value = cache.update(key, value, self.layer_idx)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        # Manual attention — all matmuls verified
        attn_scores = self.qk_mm(query, key.transpose(2, 3))  # [B, H, S, S]
        attn_scores = attn_scores * self.scaling

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask[:, :, :, : key.shape[-2]]

        dtype = query.dtype
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype)

        self.runtime.submit_elementwise(
            f"layer{self.layer_idx}.softmax", attn_scores, attn_probs, "softmax"
        )

        attn_probs = F.dropout(attn_probs, p=self.attention_dropout if self.training else 0.0, training=self.training)

        attn_output = self.kv_mm(attn_probs, value)  # [B, H, S, D]

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)

        # Qwen3.5 gated attention: attn_output *= sigmoid(gate)
        if attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(attn_gate)

        output = self.o_proj.forward(attn_output)
        output = self.o_proj.add_bias(output)

        return output, attn_probs
