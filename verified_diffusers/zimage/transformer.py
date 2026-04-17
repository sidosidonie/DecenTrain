from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from verified_diffusers.zimage.layers import VerifyLinearModule
from verified_core.runtime import VerifyRuntime
from verified_diffusers.zimage.transformer_block import VerifiedZImageTransformerBlock


def _wrap_linear(module: nn.Module, attr_name: str, runtime: VerifyRuntime, tag: str) -> None:
    linear = getattr(module, attr_name, None)
    if isinstance(linear, nn.Linear):
        setattr(module, attr_name, VerifyLinearModule(linear, runtime, tag))


class VerifiedZImageTransformer2DModel(nn.Module):
    """
    Wrap diffusers ZImageTransformer2DModel and replace core modules with verified implementations.
    """

    def __init__(self, transformer: nn.Module, runtime: VerifyRuntime):
        super().__init__()
        self.transformer = transformer
        self.runtime = runtime
        self._patch_modules()

    def _patch_modules(self) -> None:
        self.transformer.noise_refiner = nn.ModuleList(
            [
                VerifiedZImageTransformerBlock(layer, self.runtime, f"noise_refiner.{i}")
                for i, layer in enumerate(self.transformer.noise_refiner)
            ]
        )
        self.transformer.context_refiner = nn.ModuleList(
            [
                VerifiedZImageTransformerBlock(layer, self.runtime, f"context_refiner.{i}")
                for i, layer in enumerate(self.transformer.context_refiner)
            ]
        )
        self.transformer.layers = nn.ModuleList(
            [VerifiedZImageTransformerBlock(layer, self.runtime, f"layers.{i}") for i, layer in enumerate(self.transformer.layers)]
        )

        # x embedder
        for key, emb in list(self.transformer.all_x_embedder.items()):
            if isinstance(emb, nn.Linear):
                self.transformer.all_x_embedder[key] = VerifyLinearModule(emb, self.runtime, f"all_x_embedder.{key}")

        # final layer
        for key, layer in list(self.transformer.all_final_layer.items()):
            _wrap_linear(layer, "linear", self.runtime, f"final_layer.{key}.linear")
            if hasattr(layer, "adaLN_modulation") and isinstance(layer.adaLN_modulation, nn.Sequential):
                if len(layer.adaLN_modulation) > 1 and isinstance(layer.adaLN_modulation[1], nn.Linear):
                    layer.adaLN_modulation[1] = VerifyLinearModule(
                        layer.adaLN_modulation[1], self.runtime, f"final_layer.{key}.adaLN"
                    )

        # timestep embedder mlp
        if hasattr(self.transformer, "t_embedder") and hasattr(self.transformer.t_embedder, "mlp"):
            mlp = self.transformer.t_embedder.mlp
            if isinstance(mlp, nn.Sequential):
                if len(mlp) > 0 and isinstance(mlp[0], nn.Linear):
                    mlp[0] = VerifyLinearModule(mlp[0], self.runtime, "t_embedder.mlp0")
                if len(mlp) > 2 and isinstance(mlp[2], nn.Linear):
                    mlp[2] = VerifyLinearModule(mlp[2], self.runtime, "t_embedder.mlp2")

        # cap embedder
        if hasattr(self.transformer, "cap_embedder") and isinstance(self.transformer.cap_embedder, nn.Sequential):
            if len(self.transformer.cap_embedder) > 1 and isinstance(self.transformer.cap_embedder[1], nn.Linear):
                self.transformer.cap_embedder[1] = VerifyLinearModule(
                    self.transformer.cap_embedder[1], self.runtime, "cap_embedder.linear"
                )

    def flush(self) -> None:
        self.runtime.flush()

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        return_dict: bool = True,
    ):
        out = self.transformer(
            x=x,
            t=t,
            cap_feats=cap_feats,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=return_dict,
        )
        if self.runtime.config.flush_on_pipeline_end:
            self.runtime.flush()
        return out

    def __getattr__(self, name):
        # Keep compatibility with attributes accessed by pipeline internals.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)
