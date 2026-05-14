"""Functional tests for examples/multi_machine_attn_llama.py."""
from __future__ import annotations

import importlib.util
import pathlib
import socket
import struct
import sys
import threading

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


def _start_worker_thread(worker):
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    return t


def test_worker_load_round_trip():
    """Coordinator-side: send LOAD_REQ, receive LOAD_ACK."""
    m = _load()
    port = m.pick_free_port()
    cfg_kwargs = dict(hidden=32, heads=4, kv_heads=4, head_dim=8,
                      batch=2, seq=6, weight_seed=99)
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port,
                      device="cpu", inject_fault="none",
                      pipeline=False, quiet=True)
    t = _start_worker_thread(worker)
    m.wait_port(port, timeout=3.0)

    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = m.pack_load_req(
        hidden=cfg_kwargs["hidden"], heads=cfg_kwargs["heads"],
        kv_heads=cfg_kwargs["kv_heads"], head_dim=cfg_kwargs["head_dim"],
        rope_base_e9=int(500000.0 * 1e3),  # encoded as int (kHz form)
        weight_seed=cfg_kwargs["weight_seed"],
        dtype_id=m.DTYPE_FP32,
    )
    m.send_msg(sock, m.MSG_LOAD_REQ, body)
    mt, ack = m.recv_msg(sock)
    assert mt == m.MSG_LOAD_ACK
    assert m.unpack_load_ack(ack)["status"] == 0

    m.send_msg(sock, m.MSG_CLOSE, b"")
    sock.close()
    t.join(2.0)
    assert worker.hidden == cfg_kwargs["hidden"]
    assert worker.heads == cfg_kwargs["heads"]
    assert worker.kv_heads == cfg_kwargs["kv_heads"]


def test_coordinator_loopback_round_passes_with_no_fault():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=32, heads=4, kv_heads=4, head_dim=8,
                            batch=2, seq=8, wire_dtype=m.DTYPE_FP32,
                            weight_seed=7)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", pipeline=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K, pipeline=False)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is True
    assert rm.mse[m.OP_Q] < 1e-3
    assert rm.mse[m.OP_O] < 1e-3
    assert rm.bytes_recv == rm.bytes_recv_predicted


def test_coordinator_loopback_detects_flip_v_fault():
    m = _load()
    cfg = m.AttnLlamaConfig(hidden=32, heads=4, kv_heads=4, head_dim=8,
                            batch=2, seq=8, wire_dtype=m.DTYPE_FP32,
                            weight_seed=7)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="flip_v", pipeline=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K, pipeline=False)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is False
    assert rm.mse[m.OP_V] > 1e-2  # well above the 1e-3 threshold


import json
import subprocess
import tempfile


def test_loopback_subprocess_smoke():
    """Run the example end-to-end as a subprocess via --role loopback."""
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--hidden", "32", "--heads", "4", "--kv-heads", "4",
         "--head-dim", "8", "--batch", "2", "--seq", "8",
         "--wire-dtype", "fp32", "--rounds", "5", "--warmup", "1"],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}\nstdout:\n{out.stdout}"
    assert "rounds passed     4 / 4" in out.stdout
    assert "End-to-end" in out.stdout


def test_loopback_subprocess_fault_detection():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--hidden", "32", "--heads", "4", "--kv-heads", "4",
         "--head-dim", "8", "--batch", "2", "--seq", "8",
         "--wire-dtype", "fp32", "--rounds", "3", "--warmup", "0",
         "--inject-fault", "scale_o"],
        capture_output=True, timeout=60, text=True,
    )
    assert out.returncode == 0
    # All measured rounds should fail
    assert "rounds passed     0 / 3" in out.stdout


# ---------------------------------------------------------------------------
# E1: Determinism + fault-margin invariants
# ---------------------------------------------------------------------------

def _run_with_json(args: list) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)] + args + ["--json-report", path],
        capture_output=True, timeout=120, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    with open(path) as f:
        return json.load(f)


BASE_ARGS_LLAMA = [
    "--role", "loopback", "--device", "cpu",
    "--hidden", "32", "--heads", "4", "--kv-heads", "4", "--head-dim", "8",
    "--batch", "2", "--seq", "8",
    "--wire-dtype", "fp32", "--rounds", "3", "--warmup", "0",
]


def test_llama_determinism_two_runs_identical_mse():
    a = _run_with_json(BASE_ARGS_LLAMA + ["--weight-seed", "0x1234"])
    b = _run_with_json(BASE_ARGS_LLAMA + ["--weight-seed", "0x1234"])
    a_mse = [(r["request_id"], r["mse"]) for r in a["per_round"]]
    b_mse = [(r["request_id"], r["mse"]) for r in b["per_round"]]
    assert a_mse == b_mse, "MSE must be deterministic for fixed seed"


@pytest.mark.parametrize("fault,floor", [
    ("flip_v", 1e-4),
    ("scale_o", 1e-4),
    ("drop_softmax", 1e-4),
    # drop_rope produces a subtle RoPE phase shift; MSE stays ~5e-8 at dim=32
    ("drop_rope", 1e-9),
])
def test_llama_fault_margin_above_floor(fault, floor):
    rep = _run_with_json(BASE_ARGS_LLAMA + ["--inject-fault", fault])
    max_mse = max(max(float(v) for v in r["mse"].values()) for r in rep["per_round"])
    assert max_mse > floor, f"fault {fault}: max mse {max_mse:.2e} not > {floor:.0e}"
