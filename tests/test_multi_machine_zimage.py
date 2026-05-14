"""Functional tests for examples/multi_machine_zimage.py."""
from __future__ import annotations

import importlib.util
import pathlib
import socket
import subprocess
import sys
import threading

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_zimage.py"


def _load():
    spec = importlib.util.spec_from_file_location("mmzimage", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmzimage"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_op_kind_constants():
    m = _load()
    assert m.OP_Q == 1 and m.OP_K == 2 and m.OP_V == 3 and m.OP_O == 4
    assert m.OP_W1 == 5 and m.OP_W3 == 6 and m.OP_W2 == 7


def test_pack_unpack_op_tag_with_block():
    m = _load()
    # block_idx << 4 | op_kind
    tag = m.make_op_tag(block_idx=5, op_kind=m.OP_W1)
    assert tag == (5 << 4) | m.OP_W1
    block, kind = m.split_op_tag(tag)
    assert block == 5 and kind == m.OP_W1


def test_zimage_config_defaults():
    m = _load()
    cfg = m.ZimageConfig()
    assert cfg.dim == 1536
    assert cfg.heads == 12
    assert cfg.head_dim == 128
    assert cfg.ffn_inter == 4096
    assert cfg.n_layers == 12


def test_make_block_weights_shape_and_count():
    m = _load()
    cfg = m.ZimageConfig(dim=32, heads=4, head_dim=8, ffn_inter=64,
                         n_layers=2, batch=1, seq=4)
    blocks = m.make_zimage_block_weights(cfg, dtype=torch.float32, device="cpu")
    assert len(blocks) == cfg.n_layers
    block = blocks[0]
    assert block.q_proj.weight.shape == (4 * 8, 32)
    assert block.o_proj.weight.shape == (32, 4 * 8)
    assert block.w1.weight.shape == (64, 32)
    assert block.w2.weight.shape == (32, 64)
    assert block.w3.weight.shape == (64, 32)
    assert block.attention_norm1.shape == (32,)
    assert block.ffn_norm1.shape == (32,)


def test_make_block_weights_per_block_unique():
    m = _load()
    cfg = m.ZimageConfig(dim=16, heads=2, head_dim=8, ffn_inter=32,
                         n_layers=2, batch=1, seq=2, weight_seed=1)
    blocks = m.make_zimage_block_weights(cfg, dtype=torch.float32, device="cpu")
    # Block 0 and block 1 must have DIFFERENT q_proj weights
    assert not torch.equal(blocks[0].q_proj.weight, blocks[1].q_proj.weight)
