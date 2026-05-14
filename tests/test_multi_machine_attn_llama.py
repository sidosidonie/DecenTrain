"""Functional tests for examples/multi_machine_attn_llama.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch


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
