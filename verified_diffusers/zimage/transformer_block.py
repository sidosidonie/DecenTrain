from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from diffusers.models.transformers.transformer_z_image import select_per_token
except ImportError:
    def select_per_token(noisy, clean, noise_mask, seq_len):
        mask = noise_mask[:, :seq_len].unsqueeze(-1)
        return torch.where(mask, noisy[:, :seq_len], clean[:, :seq_len])

from verified_diffusers.zimage.attention import VerifiedZImageAttention
from verified_diffusers.zimage.layers import VerifyLinearModule
from verified_diffusers.zimage.mlp import VerifiedZImageFeedForward
from verified_core.runtime import VerifyRuntime


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
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]

            if noise_mask is not None:
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
                gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
            else:
                assert adaln_input is not None
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
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
