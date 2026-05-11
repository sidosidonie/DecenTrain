import sys
from pathlib import Path

import pytest
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import FeedForward, ZSingleStreamAttnProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_diffusers.zimage.attention import VerifiedZImageAttention
from verified_core.config import VerifyConfig
from verified_diffusers.zimage.mlp import VerifiedZImageFeedForward
from verified_core.runtime import VerifyRuntime


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


# ---------------------------------------------------------------------------
# Chain pattern: explicit input_state_key (multi-machine seed path)
#
# These tests exercise the path where the caller pre-populates cpu_state with
# the chain input and passes its key to forward(). On the multi-machine
# coordinator, the input is reconstructed via CPU recompute (never shipped),
# and the chain reads from cpu_state — never from a D2H of the GPU input.
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_verified_zimage_mlp_uses_explicit_cpu_state_key():
    """When input_state_key is provided, the chain reads from cpu_state.

    Proof: pre-populate cpu_state with a CORRUPTED input. If the chain reads
    from cpu_state, SLALOM will fail (loss above threshold). If the chain
    silently falls back to a GPU D2H, it would pass — so a failure here is
    the signal we want.
    """
    runtime = _make_runtime()
    ff = FeedForward(dim=64, hidden_dim=128).to("cuda")
    ff.eval()
    verified = VerifiedZImageFeedForward(ff, runtime, "unit.mlp").to("cuda")

    x = torch.randn(2, 24, 64, device="cuda", dtype=torch.float32)
    # Seed cpu_state with a corrupted input (random noise of same shape).
    bad_input_cpu = torch.randn_like(x, device="cpu", dtype=torch.float32)
    runtime.cpu_state_set("explicit.mlp.input", bad_input_cpu)

    verified(x, input_state_key="explicit.mlp.input")

    with pytest.raises(RuntimeError):
        runtime.flush()
    runtime.shutdown()


@torch.no_grad()
def test_verified_zimage_mlp_passes_with_correct_cpu_state_seed():
    """When cpu_state holds the correct input, the chain passes."""
    runtime = _make_runtime()
    ff = FeedForward(dim=64, hidden_dim=128).to("cuda")
    ff.eval()
    verified = VerifiedZImageFeedForward(ff, runtime, "unit.mlp").to("cuda")

    x = torch.randn(2, 24, 64, device="cuda", dtype=torch.float32)
    # Seed with the correct CPU value of x.
    runtime.cpu_state_set("explicit.mlp.input", x.detach().cpu().float())

    y_ref = ff(x)
    y_ver = verified(x, input_state_key="explicit.mlp.input")
    runtime.flush()

    assert torch.allclose(y_ref, y_ver, atol=1e-4, rtol=1e-4)
    assert runtime.pending_tasks == 0
    runtime.shutdown()


@torch.no_grad()
def test_verified_zimage_attention_uses_explicit_cpu_state_key():
    """Same chain-source proof for attention: corrupted seed -> SLALOM fails."""
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

    bad_input_cpu = torch.randn_like(hidden_states, device="cpu", dtype=torch.float32)
    runtime.cpu_state_set("explicit.attn.input", bad_input_cpu)

    verified(
        hidden_states,
        attention_mask=attention_mask,
        freqs_cis=freqs,
        input_state_key="explicit.attn.input",
    )

    with pytest.raises(RuntimeError):
        runtime.flush()
    runtime.shutdown()
