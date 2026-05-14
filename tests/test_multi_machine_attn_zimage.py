"""Functional tests for examples/multi_machine_attn_zimage.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import threading

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_attn_zimage.py"


def _load():
    spec = importlib.util.spec_from_file_location("mmattnzi", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmattnzi"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_op_tag_constants():
    m = _load()
    assert m.OP_Q == 1 and m.OP_K == 2 and m.OP_V == 3 and m.OP_O == 4


def test_attn_zimage_config_defaults():
    m = _load()
    cfg = m.AttnZimageConfig()
    assert cfg.dim == 1536
    assert cfg.heads == 12
    assert cfg.head_dim == 128
    assert cfg.qk_norm == "rms"


def test_make_zimage_weights_shapes():
    m = _load()
    cfg = m.AttnZimageConfig(dim=64, heads=4, head_dim=16, batch=2, seq=8)
    q, k, v, o, nq, nk = m.make_zimage_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    assert q.weight.shape == (4 * 16, 64)
    assert k.weight.shape == (4 * 16, 64)
    assert v.weight.shape == (4 * 16, 64)
    assert o.weight.shape == (64, 4 * 16)
    assert nq.shape == (16,)   # per head_dim RMSNorm scale
    assert nk.shape == (16,)


def test_make_zimage_weights_norm_skipped_when_none():
    m = _load()
    cfg = m.AttnZimageConfig(dim=64, heads=4, head_dim=16, batch=2, seq=8,
                             qk_norm="none")
    q, k, v, o, nq, nk = m.make_zimage_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    assert nq is None and nk is None


def test_compute_zimage_attn_forward_matches_reference():
    m = _load()
    cfg = m.AttnZimageConfig(dim=32, heads=4, head_dim=8, batch=2, seq=6,
                             qk_norm="rms", weight_seed=3)
    q_proj, k_proj, v_proj, o_proj, nq, nk = m.make_zimage_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    freqs = m.precompute_zimage_freqs_cis(cfg.head_dim, cfg.seq, theta=cfg.rope_theta)

    torch.manual_seed(0)
    x = torch.randn(cfg.batch, cfg.seq, cfg.dim)
    q_raw, k_raw, v_raw, o_raw = m.compute_zimage_attn_forward(
        x, q_proj, k_proj, v_proj, o_proj, nq, nk, freqs, cfg)

    # Reference
    q = q_proj(x).unflatten(-1, (cfg.heads, cfg.head_dim))
    k = k_proj(x).unflatten(-1, (cfg.heads, cfg.head_dim))
    v = v_proj(x).unflatten(-1, (cfg.heads, cfg.head_dim))
    eps = m.ZIMAGE_QK_NORM_EPS
    q = m.rmsnorm_cpu(q, nq, eps, scale_offset=0.0)
    k = m.rmsnorm_cpu(k, nk, eps, scale_offset=0.0)
    q = m.apply_rotary_emb_zimage(q, freqs)
    k = m.apply_rotary_emb_zimage(k, freqs)
    q_t = q.permute(0, 2, 1, 3); k_t = k.permute(0, 2, 1, 3); v_t = v.permute(0, 2, 1, 3)
    scores = q_t @ k_t.transpose(2, 3) * (cfg.head_dim ** -0.5)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ v_t).permute(0, 2, 1, 3).flatten(2, 3)
    o_ref = o_proj(attn_out)

    assert torch.allclose(q_raw, q_proj(x), atol=1e-5)
    assert torch.allclose(o_raw, o_ref, atol=1e-4)
