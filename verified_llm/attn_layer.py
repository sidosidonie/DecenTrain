from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
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
    def __init__(
        self,
        llama: LlamaAttention,
        runtime: VerifyRuntime,
        tag_prefix: str = "",
        noise=None,
    ):
        super().__init__()
        self.config = llama.config
        self.layer_idx = llama.layer_idx
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

        prefix = tag_prefix or f"layer{self.layer_idx}"
        self.q_proj = VerifyLinear(llama.q_proj, runtime, f"{prefix}.q", noise)
        self.k_proj = VerifyLinear(llama.k_proj, runtime, f"{prefix}.k", noise)
        self.v_proj = VerifyLinear(llama.v_proj, runtime, f"{prefix}.v", noise)
        self.o_proj = VerifyLinear(llama.o_proj, runtime, f"{prefix}.o", noise)
        self.qk_mm = VerifyMatmul(runtime, f"{prefix}.qk")
        self.kv_mm = VerifyMatmul(runtime, f"{prefix}.kv")
        self.runtime = runtime

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # Projections — async verified via VerifyRuntime
        query = self.q_proj.forward(hidden_states)
        query = self.q_proj.add_bias(query)
        query = query.view(hidden_shape).transpose(1, 2)

        key = self.k_proj.forward(hidden_states)
        key = self.k_proj.add_bias(key)
        key = key.view(hidden_shape).transpose(1, 2)

        value = self.v_proj.forward(hidden_states)
        value = self.v_proj.add_bias(value)
        value = value.view(hidden_shape).transpose(1, 2)

        # Rotary embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_value.update(key, value, self.layer_idx, cache_kwargs)

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

        output = self.o_proj.forward(attn_output)
        output = self.o_proj.add_bias(output)

        return output, attn_probs
