import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import FluxAttnProcessor2_0, Attention
from nltk.corpus.reader.markdown import List
from verified_diffusers.hooks import *
from verified_llm.verify_linear import SyncVerifyLinear, all_matmul
from verified_llm.utils.log_utils import g_logger
from diffusers import FluxPipeline, DiffusionPipeline, ZImagePipeline
from diffusers.models.attention_dispatch import _AttentionBackendRegistry, _check_device, _check_shape, AttentionBackendName
from pprint import pprint
from typing import Optional
import math
import torch.profiler as profiler
import time

import pandas as pd
from fire import Fire
from typing import OrderedDict
from collections import defaultdict, OrderedDict

torch.set_num_threads(10)
torch.set_num_interop_threads(2)

ZSingleStreamAttnProcessor._attention_backend =  AttentionBackendName._NATIVE_VERIFY


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

    # print(f"{query.shape=}")
    # print(f"{key.shape=}")
    # print(f"{value.shape=}")

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_VERIFY,
    constraints=[_check_device, _check_shape],
    supports_context_parallel=True,
)
def native_verify(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config = None 
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native attention backend does not support setting `return_lse=True`.")

    assert _parallel_config is None

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    # print(f"{query.shape=}")
    # print(f"{key.shape=}")
    # print(f"{value.shape=}")

    out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)
    return out


def replace_linear_for_pipeline(pipe):
    replace_linear(pipe.transformer)

def test_model(model_name, prompt):

    pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image", dtype=torch.bfloat16, device_map="cuda")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():

        hook_ori = ModelInfoHook(pipe.transformer)
        torch.cuda.synchronize()
        starter.record()

        # with profiler.profile(
        #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=False
        # ) as prof:
        image = pipe(prompt, num_inference_steps=10).images[0]

        # image = pipe(
        #     prompt,
        #     guidance_scale=3.5,
        #     num_inference_steps=30
        # ).images[0]

        ender.record()
        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender)
        print(f"Inference time: {elapsed_ms:.2f} ms")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        gpu_stream = torch.cuda.default_stream()
        cpu_stream = torch.cuda.Stream()
        replace_linear(pipe.transformer, gpu_stream, cpu_stream)
        hook_ver = ModelInfoHook(pipe.transformer)
        image = pipe(prompt, num_inference_steps=10).images[0]

        # image.save("z-image-verified.png")

        return to_df(hook_ori.model_info, hook_ver.model_info)


prompts = {
    "short": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "long": """Ultra-high-resolution iPhone wallpaper, vertical 9:16, 4K+.
Soft beige and warm cream gradient, smooth color transitions.
Minimalist luxury aesthetic, calm and elegant.
Clean composition with generous negative space at the top for lock-screen clock.
No text, no logos, no watermark.
Apple-inspired, premium digital art, commercial-ready."""
}

models = {
    "flux" : "black-forest-labs/FLUX.1-schnell",
    "zimage" : "Tongyi-MAI/Z-Image"
}

def main(model_name="zimage", prompt="long"):
    df = test_model(models[model_name], prompt)
    df.to_csv("time.csv")


if __name__ == "__main__":
    Fire(main)