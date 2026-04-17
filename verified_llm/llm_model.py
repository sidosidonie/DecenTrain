"""
Build LLM models with verified linear layers using VerifyRuntime.

Supports any HuggingFace causal LM with standard attention/MLP structure:
- Attention: modules with q_proj, k_proj, v_proj, o_proj (Llama, Qwen, Mistral, etc.)
- MLP: modules with gate_proj, up_proj, down_proj (Llama, Qwen, Mistral, etc.)

CPU chain design:
  - Only matmul OUTPUTS are D2H copied from GPU.
  - CPU maintains its own state across layers (via runtime cpu_state).
  - All non-linear operations (layernorm, RoPE, softmax, SiLU, residual)
    are computed on CPU from chain data.
  - Initial hidden_states (embedding output) is D2H'd once per forward pass.
  - Non-verified attention layers bridge via D2H of post-attention state.
"""
from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from verified_core.config import VerifyConfig
from verified_core.runtime import VerifyRuntime
from verified_llm.attn_layer import LlamaAttentionVerify
from verified_llm.mlp_layer import LlamaMLPVerify


def _is_attention_module(module) -> bool:
    """Detect attention modules by duck typing (q/k/v/o_proj attributes)."""
    if module is None:
        return False
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
        and isinstance(module.q_proj, nn.Linear)
    )


def _is_mlp_module(module) -> bool:
    """Detect MLP modules by duck typing (gate/up/down_proj attributes)."""
    if module is None:
        return False
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_proj, nn.Linear)
    )


def _get_layernorm_params(norm_module: nn.Module):
    """Extract weight and eps from a layernorm module (CPU, float32)."""
    weight = norm_module.weight.detach().float().cpu()
    eps = getattr(norm_module, "eps", None)
    if eps is None:
        eps = getattr(norm_module, "variance_epsilon", 1e-6)
    return weight, eps


def _register_initial_d2h_hook(model: nn.Module, runtime: VerifyRuntime) -> None:
    """Hook the first decoder layer to D2H the initial hidden_states.

    Each forward pass:
      1. Clear CPU state from the previous pass.
      2. Async-copy the embedding output to CPU as ``layer_input_0``.
    """
    first_layer = model.model.layers[0]

    def _hook(module, args):
        # Clear stale CPU state from the previous forward pass.
        # (flush() also clears, but this is a safety net when flush is
        # not called between consecutive forwards, e.g. during warmup.)
        runtime.cpu_state_clear()

        hidden_states = args[0]
        cpu_h, done_evt = runtime.d2h_async(hidden_states)
        runtime.cpu_state_set_d2h("layer_input_0", cpu_h, done_evt)

    first_layer.register_forward_pre_hook(_hook)


def _register_bridge_hook(
    layer: nn.Module, runtime: VerifyRuntime, layer_idx: int,
) -> None:
    """For non-verified attention layers: D2H the post-attention state.

    The hook captures the input to ``post_attention_layernorm``, which is
    ``hidden_states`` after the attention module + residual connection.
    This lets the MLP chain read its input from CPU state.
    """
    post_ln = layer.post_attention_layernorm

    def _hook(module, args):
        hidden_states = args[0]
        cpu_h, done_evt = runtime.d2h_async(hidden_states)
        runtime.cpu_state_set_d2h(f"post_attn_{layer_idx}", cpu_h, done_evt)

    post_ln.register_forward_pre_hook(_hook)


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
        n_attn = 0
        n_mlp = 0

        for idx, layer in enumerate(model.model.layers):
            # ── Layernorm weights for CPU chain ──
            input_ln_weight, input_ln_eps = _get_layernorm_params(
                layer.input_layernorm,
            )
            post_attn_ln_weight, post_attn_ln_eps = _get_layernorm_params(
                layer.post_attention_layernorm,
            )

            # ── Replace attention (if standard q/k/v/o_proj module) ──
            attn_module = getattr(layer, "self_attn", None)
            if _is_attention_module(attn_module):
                attn_layer_idx = getattr(attn_module, "layer_idx", idx)
                verified_attn = LlamaAttentionVerify(
                    attn_module,
                    runtime,
                    tag_prefix=f"layer{attn_layer_idx}",
                    noise=noise,
                    layer_idx=idx,
                    input_ln_weight=input_ln_weight,
                    input_ln_eps=input_ln_eps,
                )
                layer.self_attn = verified_attn
                n_attn += 1
            else:
                # Non-verified attention (e.g. linear_attention):
                # bridge D2H so the MLP chain can read post-attn state.
                _register_bridge_hook(layer, runtime, idx)

            # ── Replace MLP ──
            mlp_module = layer.mlp
            if _is_mlp_module(mlp_module):
                verified_mlp = LlamaMLPVerify(
                    mlp_module,
                    runtime,
                    tag_prefix=f"mlp.layer{idx}",
                    noise_scale=noise,
                    layer_idx=idx,
                    post_attn_ln_weight=post_attn_ln_weight,
                    post_attn_ln_eps=post_attn_ln_eps,
                )
                layer.mlp = verified_mlp
                n_mlp += 1

        # Hook to D2H initial hidden_states at each forward pass
        _register_initial_d2h_hook(model, runtime)

        print(f"[verified] Replaced {n_attn} attention + {n_mlp} MLP modules")

    model._verify_runtime = runtime
    return model


if __name__ == "__main__":
    mod = create_llm_model("meta-llama/Llama-3.2-1B-Instruct", verify=True)
    print(mod)
