"""Functional tests for examples/multi_machine_attn_llama.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_attn_llama.py"


def _load():
    spec = importlib.util.spec_from_file_location("mmattnllama", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmattnllama"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_op_tag_constants():
    m = _load()
    assert m.OP_Q == 1
    assert m.OP_K == 2
    assert m.OP_V == 3
    assert m.OP_O == 4


def test_attn_llama_config_defaults():
    m = _load()
    cfg = m.AttnLlamaConfig()
    assert cfg.hidden == 4096
    assert cfg.heads == 32
    assert cfg.kv_heads == 32
    assert cfg.head_dim == 128
    assert cfg.batch == 1
    assert cfg.seq == 512
    assert cfg.rope_base == 500000.0
    assert cfg.weight_seed == 0xC0FFEE


def test_make_attn_weights_shapes_mha():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=64, heads=8, kv_heads=8, head_dim=8,
                            batch=1, seq=4)
    q, k, v, o = m.make_attn_weights(cfg, dtype=torch.float32, device="cpu")
    assert q.weight.shape == (8 * 8, 64)
    assert k.weight.shape == (8 * 8, 64)
    assert v.weight.shape == (8 * 8, 64)
    assert o.weight.shape == (64, 8 * 8)
    assert all(p.bias is None for p in (q, k, v, o))


def test_make_attn_weights_shapes_gqa():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=64, heads=8, kv_heads=2, head_dim=8,
                            batch=1, seq=4)
    q, k, v, o = m.make_attn_weights(cfg, dtype=torch.float32, device="cpu")
    assert q.weight.shape == (8 * 8, 64)        # heads * head_dim
    assert k.weight.shape == (2 * 8, 64)        # kv_heads * head_dim
    assert v.weight.shape == (2 * 8, 64)
    assert o.weight.shape == (64, 8 * 8)


def test_make_attn_weights_deterministic_across_dtype():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=64, heads=8, kv_heads=8, head_dim=8,
                            batch=1, seq=4, weight_seed=42)
    q32, k32, v32, o32 = m.make_attn_weights(cfg, torch.float32, "cpu")
    q16, k16, v16, o16 = m.make_attn_weights(cfg, torch.float16, "cpu")
    assert torch.allclose(q32.weight.half().float(), q16.weight.float())
    assert torch.allclose(o32.weight.half().float(), o16.weight.float())


def test_repeat_kv_helper():
    m = _load()
    x = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).float()
    out = m.repeat_kv(x, n_rep=3)
    assert out.shape == (2, 6, 3, 4)
    # Each kv head should be replicated n_rep times consecutively
    assert torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 1], out[:, 2])
    assert torch.equal(out[:, 3], out[:, 4])


def test_worker_forward_matches_reference_compute():
    """Compute the worker's q/k/v/o tensors directly and check against a
    plain torch reference path with RoPE + causal softmax."""
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=32, heads=4, kv_heads=4, head_dim=8,
                            batch=2, seq=6, weight_seed=1)
    q_proj, k_proj, v_proj, o_proj = m.make_attn_weights(
        cfg, dtype=torch.float32, device="cpu")
    cos, sin = m.precompute_rope_cos_sin(cfg.head_dim, cfg.seq, cfg.rope_base)

    torch.manual_seed(0)
    x = torch.randn(cfg.batch, cfg.seq, cfg.hidden)
    q_raw, k_raw, v_raw, o_raw = m.compute_attn_forward(
        x, q_proj, k_proj, v_proj, o_proj, cos, sin, cfg)

    # Reference path
    q = q_proj(x).view(cfg.batch, cfg.seq, cfg.heads, cfg.head_dim).transpose(1, 2)
    k = k_proj(x).view(cfg.batch, cfg.seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    v = v_proj(x).view(cfg.batch, cfg.seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    q_rope, k_rope = m.apply_rope_llama(q, k, cos, sin)
    k_rep = m.repeat_kv(k_rope, cfg.num_kv_groups)
    v_rep = m.repeat_kv(v, cfg.num_kv_groups)
    scores = q_rope @ k_rep.transpose(-2, -1) * (cfg.head_dim ** -0.5)
    causal = torch.triu(torch.full((cfg.seq, cfg.seq), float("-inf")), diagonal=1)
    scores = scores + causal
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    attn_out = (probs @ v_rep).transpose(1, 2).reshape(cfg.batch, cfg.seq, cfg.hidden)
    o_ref = o_proj(attn_out)

    assert torch.allclose(q_raw, q_proj(x), atol=1e-5)
    assert torch.allclose(k_raw, k_proj(x), atol=1e-5)
    assert torch.allclose(v_raw, v_proj(x), atol=1e-5)
    assert torch.allclose(o_raw, o_ref, atol=1e-4)
