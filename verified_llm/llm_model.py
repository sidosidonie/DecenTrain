"""
Build LLM models with verified linear layers using VerifyRuntime.

Supports any HuggingFace causal LM with standard attention/MLP structure:
- Attention: modules with q_proj, k_proj, v_proj, o_proj (Llama, Qwen, Mistral, etc.)
- MLP: modules with gate_proj, up_proj, down_proj (Llama, Qwen, Mistral, etc.)
"""
from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.attn_layer import LlamaAttentionVerify
from verified_llm.mlp_layer import LlamaMLPVerify


def _is_attention_module(module: nn.Module) -> bool:
    """Detect attention modules by duck typing (q/k/v/o_proj attributes)."""
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
        and isinstance(module.q_proj, nn.Linear)
    )


def _is_mlp_module(module: nn.Module) -> bool:
    """Detect MLP modules by duck typing (gate/up/down_proj attributes)."""
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_proj, nn.Linear)
    )


def replace_attn(model: nn.Module, runtime: VerifyRuntime, noise=None) -> int:
    """Replace all compatible attention modules. Returns count replaced."""
    count = 0
    for name, module in model.named_children():
        if _is_attention_module(module):
            layer_idx = getattr(module, "layer_idx", name)
            verified = LlamaAttentionVerify(
                module, runtime, tag_prefix=f"layer{layer_idx}", noise=noise
            )
            setattr(model, name, verified)
            count += 1
        else:
            count += replace_attn(module, runtime, noise)
    return count


def replace_mlp(model: nn.Module, runtime: VerifyRuntime, noise=None) -> int:
    """Replace all compatible MLP modules. Returns count replaced."""
    count = 0
    for name, module in model.named_children():
        if _is_mlp_module(module):
            verified = LlamaMLPVerify(
                module, runtime, tag_prefix=f"mlp.{name}", noise_scale=noise
            )
            setattr(model, name, verified)
            count += 1
        else:
            count += replace_mlp(module, runtime, noise)
    return count


def dump_layer_outputs(model):
    layer_outputs = {}

    def hook_fn(name, module, input, output):
        layer_outputs[name] = output

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            hooks.append(module.register_forward_hook(functools.partial(hook_fn, name)))
    return layer_outputs, hooks


def create_llm_model(
    model_path: str,
    verify: bool = False,
    config: Optional[VerifyConfig] = None,
    noise=None,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    **from_pretrained_kwargs,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, **from_pretrained_kwargs
    )
    # Force eager attention so we can intercept Q@K and attn@V matmuls.
    # Handle nested text_config (e.g. Qwen3.5 multimodal models).
    text_cfg = getattr(model.config, "text_config", model.config)
    text_cfg._attn_implementation = "eager"
    model.config._attn_implementation = "eager"
    model.to(device)

    runtime = None
    if verify:
        runtime = VerifyRuntime(config or VerifyConfig())
        n_attn = replace_attn(model, runtime, noise)
        n_mlp = replace_mlp(model, runtime, noise)
        print(f"[verified] Replaced {n_attn} attention + {n_mlp} MLP modules")

    model._verify_runtime = runtime
    return model


if __name__ == "__main__":
    mod = create_llm_model("meta-llama/Llama-3.2-1B-Instruct", verify=True)
    print(mod)
