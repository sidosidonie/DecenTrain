from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from verified_diffusers.zimage.attention import VerifiedZImageAttention
from verified_diffusers.zimage.layers import VerifyLinearModule
from verified_diffusers.zimage.mlp import VerifiedZImageFeedForward
from verified_diffusers.zimage.runtime import VerifyRuntime


class VerifiedZImageTransformerBlock(nn.Module):
    def __init__(self, block: nn.Module, runtime: VerifyRuntime, tag_prefix: str):
        super().__init__()
        self.runtime = runtime
        self.dim = block.dim
        self.head_dim = block.head_dim
        self.layer_id = block.layer_id
        self.modulation = block.modulation

        self.attention = VerifiedZImageAttention(block.attention, runtime, f"{tag_prefix}.attention")
        self.feed_forward = VerifiedZImageFeedForward(block.feed_forward, runtime, f"{tag_prefix}.feed_forward")

        self.attention_norm1 = block.attention_norm1
        self.ffn_norm1 = block.ffn_norm1
        self.attention_norm2 = block.attention_norm2
        self.ffn_norm2 = block.ffn_norm2
        adaln_seq = getattr(block, "adaLN_modulation", None)
        if adaln_seq is not None:
            for i, layer in enumerate(adaln_seq):
                if isinstance(layer, nn.Linear):
                    adaln_seq[i] = VerifyLinearModule(layer, runtime, f"{tag_prefix}.adaLN.{i}")
        self.adaLN_modulation = adaln_seq

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        if self.runtime.config.flush_each_layer:
            self.runtime.flush()
        return x
