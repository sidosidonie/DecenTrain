import sys
from pathlib import Path

import pytest
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import FeedForward, ZSingleStreamAttnProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_diffusers.zimage.attention import VerifiedZImageAttention
from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.mlp import VerifiedZImageFeedForward
from verified_diffusers.zimage.runtime import VerifyRuntime


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_runtime():
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=6,
        mse_threshold=1e-4,
        max_workers=2,
        profile_enabled=True,
        fail_on_error=True,
        flush_each_layer=False,
        flush_on_pipeline_end=True,
    )
    return VerifyRuntime(cfg)


@torch.no_grad()
def test_verified_zimage_attention_matches_origin():
    runtime = _make_runtime()
    attn = Attention(
        query_dim=64,
        cross_attention_dim=None,
        dim_head=16,
        heads=4,
        qk_norm="rms_norm",
        eps=1e-5,
        bias=False,
        out_bias=False,
        processor=ZSingleStreamAttnProcessor(),
    ).to("cuda")
    attn.eval()
    verified = VerifiedZImageAttention(attn, runtime, "unit.attention").to("cuda")

    hidden_states = torch.randn(2, 32, 64, device="cuda", dtype=torch.float32)
    attention_mask = torch.ones(2, 32, device="cuda", dtype=torch.bool)
    phase = torch.randn(2, 32, 8, device="cuda", dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)

    y_ref = attn(hidden_states, attention_mask=attention_mask, freqs_cis=freqs)
    y_ver = verified(hidden_states, attention_mask=attention_mask, freqs_cis=freqs)
    runtime.flush()

    assert y_ref.shape == y_ver.shape
    assert torch.allclose(y_ref, y_ver, atol=1e-4, rtol=1e-4)
    assert runtime.pending_tasks == 0
    runtime.shutdown()


@torch.no_grad()
def test_verified_zimage_mlp_matches_origin():
    runtime = _make_runtime()
    ff = FeedForward(dim=64, hidden_dim=128).to("cuda")
    ff.eval()
    verified = VerifiedZImageFeedForward(ff, runtime, "unit.mlp").to("cuda")

    x = torch.randn(2, 24, 64, device="cuda", dtype=torch.float32)
    y_ref = ff(x)
    y_ver = verified(x)
    runtime.flush()

    assert y_ref.shape == y_ver.shape
    assert torch.allclose(y_ref, y_ver, atol=1e-4, rtol=1e-4)
    assert runtime.pending_tasks == 0
    runtime.shutdown()
