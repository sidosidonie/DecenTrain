"""
Build LLM models with verified linear layers using VerifyRuntime.
"""
from __future__ import annotations

import functools
from typing import Optional

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.attn_layer import LlamaAttentionVerify
from verified_llm.mlp_layer import LlamaMLPVerify


def replace_attn(model, runtime: VerifyRuntime, noise=None):
    for name, module in model.named_children():
        if isinstance(module, LlamaAttention):
            layer_idx = getattr(module, "layer_idx", name)
            verified = LlamaAttentionVerify(
                module, runtime, tag_prefix=f"layer{layer_idx}", noise=noise
            )
            setattr(model, name, verified)
        else:
            replace_attn(module, runtime, noise)


def replace_mlp(model, runtime: VerifyRuntime, noise=None):
    for name, module in model.named_children():
        if isinstance(module, LlamaMLP):
            # Try to infer layer index from parent naming
            verified = LlamaMLPVerify(module, runtime, tag_prefix=f"mlp.{name}", noise_scale=noise)
            setattr(model, name, verified)
        else:
            replace_mlp(module, runtime, noise)


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
):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config._attn_implementation = "eager"
    model.to("cuda")

    runtime = None
    if verify:
        runtime = VerifyRuntime(config or VerifyConfig())
        replace_attn(model, runtime, noise)
        replace_mlp(model, runtime, noise)

    model._verify_runtime = runtime
    return model


if __name__ == "__main__":
    mod = create_llm_model("meta-llama/Llama-3.2-1B-Instruct", verify=True)
    print(mod)
