from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaMLP

from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.verify_linear import VerifyLinear


class LlamaMLPVerify(nn.Module):
    def __init__(
        self,
        origin: LlamaMLP,
        runtime: VerifyRuntime,
        tag_prefix: str = "mlp",
        noise_scale=None,
    ):
        super().__init__()
        self.gate_proj = VerifyLinear(origin.gate_proj, runtime, f"{tag_prefix}.gate", noise_scale)
        self.up_proj = VerifyLinear(origin.up_proj, runtime, f"{tag_prefix}.up", noise_scale)
        self.down_proj = VerifyLinear(origin.down_proj, runtime, f"{tag_prefix}.down", noise_scale)
        self.runtime = runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        gate = self.gate_proj.add_bias(gate)
        gate = F.silu(gate)

        up = self.up_proj.forward(x)
        up = self.up_proj.add_bias(up)

        down_input = gate * up
        down = self.down_proj.forward(down_input)
        down = self.down_proj.add_bias(down)
        return down
