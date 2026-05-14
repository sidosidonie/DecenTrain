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


def test_compute_zimage_block_forward_matches_reference():
    m = _load()
    cfg = m.ZimageConfig(dim=16, heads=2, head_dim=8, ffn_inter=32,
                         n_layers=2, batch=1, seq=4, weight_seed=42)
    blocks = m.make_zimage_block_weights(cfg, torch.float32, "cpu")
    freqs = m.precompute_zimage_freqs_cis(cfg.head_dim, cfg.seq, theta=cfg.rope_theta)

    torch.manual_seed(0)
    x_in = torch.randn(cfg.batch, cfg.seq, cfg.dim)

    # Run the worker compute
    block_outs, x_out = m.compute_zimage_stack_forward(x_in, blocks, freqs, cfg)
    assert len(block_outs) == cfg.n_layers
    # Each block_outs[b] is dict with keys OP_Q/OP_K/OP_V/OP_O/OP_W1/OP_W3/OP_W2
    keys = {m.OP_Q, m.OP_K, m.OP_V, m.OP_O, m.OP_W1, m.OP_W3, m.OP_W2}
    assert set(block_outs[0].keys()) == keys

    # Reference: re-run block by block using the same compute paths
    x = x_in
    for b in range(cfg.n_layers):
        bw = blocks[b]
        x_n = m.rmsnorm_cpu(x, bw.attention_norm1, m.ZIMAGE_LAYER_NORM_EPS)
        # attention sub-block
        q = bw.q_proj(x_n); k = bw.k_proj(x_n); v = bw.v_proj(x_n)
        qh = q.unflatten(-1, (cfg.heads, cfg.head_dim))
        kh = k.unflatten(-1, (cfg.heads, cfg.head_dim))
        vh = v.unflatten(-1, (cfg.heads, cfg.head_dim))
        if bw.norm_q is not None:
            qh = m.rmsnorm_cpu(qh, bw.norm_q, m.ZIMAGE_QK_NORM_EPS)
            kh = m.rmsnorm_cpu(kh, bw.norm_k, m.ZIMAGE_QK_NORM_EPS)
        qh = m.apply_rotary_emb_zimage(qh, freqs)
        kh = m.apply_rotary_emb_zimage(kh, freqs)
        qt = qh.permute(0, 2, 1, 3); kt = kh.permute(0, 2, 1, 3); vt = vh.permute(0, 2, 1, 3)
        scores = qt @ kt.transpose(2, 3) * (cfg.head_dim ** -0.5)
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn_out = (probs @ vt).permute(0, 2, 1, 3).flatten(2, 3)
        o = bw.o_proj(attn_out)
        x_after = x + o

        # FFN sub-block
        h = m.rmsnorm_cpu(x_after, bw.ffn_norm1, m.ZIMAGE_LAYER_NORM_EPS)
        w1o = bw.w1(h); w3o = bw.w3(h)
        gated = torch.nn.functional.silu(w1o) * w3o
        w2o = bw.w2(gated)
        x = x_after + w2o

    assert torch.allclose(x_out, x, atol=1e-4)


def test_zimage_worker_load_round_trip():
    m = _load()
    cfg_kwargs = dict(dim=16, heads=2, head_dim=8, ffn_inter=32, n_layers=2,
                      batch=1, seq=4, weight_seed=7)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", fault_block=0, stream=False,
                      quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = m.pack_load_req(
        dim=cfg_kwargs["dim"], heads=cfg_kwargs["heads"],
        head_dim=cfg_kwargs["head_dim"], ffn_inter=cfg_kwargs["ffn_inter"],
        n_layers=cfg_kwargs["n_layers"],
        weight_seed=cfg_kwargs["weight_seed"],
        rope_theta_e3=int(10000.0 * 1000),
        qk_norm_id=1, dtype_id=m.DTYPE_FP32,
    )
    m.send_msg(sock, m.MSG_LOAD_REQ, body)
    mt, ack = m.recv_msg(sock)
    assert mt == m.MSG_LOAD_ACK
    assert m.unpack_load_ack(ack)["status"] == 0
    m.send_msg(sock, m.MSG_CLOSE, b"")
    sock.close()
    t.join(2.0)
    assert worker.n_layers == cfg_kwargs["n_layers"]
    assert len(worker.blocks) == cfg_kwargs["n_layers"]


def test_zimage_loopback_round_passes_with_no_fault():
    m = _load()
    cfg = m.ZimageConfig(dim=16, heads=2, head_dim=8, ffn_inter=32,
                         n_layers=2, batch=1, seq=4,
                         wire_dtype=m.DTYPE_FP32, weight_seed=11)
    port = m.pick_free_port()
    worker = m.Worker(bind_host="127.0.0.1", bind_port=port, device="cpu",
                      inject_fault="none", stream=False, quiet=True)
    t = threading.Thread(target=worker.serve_once, daemon=True)
    t.start()
    m.wait_port(port, timeout=3.0)

    coord = m.Coordinator(host="127.0.0.1", port=port, config=cfg,
                          threshold=1e-3, k=m.SLALOM_K)
    try:
        coord.connect_and_load()
        rm = coord.run_round(request_id=1, input_seed=1234)
    finally:
        coord.close()
    t.join(2.0)

    assert rm.ok is True
    assert rm.bytes_recv == rm.bytes_recv_predicted
    # Per-block per-op MSE all under threshold
    for b in range(cfg.n_layers):
        for kind in (m.OP_Q, m.OP_K, m.OP_V, m.OP_O, m.OP_W1, m.OP_W3, m.OP_W2):
            tag = m.make_op_tag(b, kind)
            assert rm.mse[tag] < 1e-3, f"block {b} kind {kind} mse={rm.mse[tag]}"


def test_zimage_loopback_subprocess_smoke():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--dim", "16", "--heads", "2", "--head-dim", "8",
         "--ffn-inter", "32", "--n-layers", "2",
         "--batch", "1", "--seq", "4",
         "--wire-dtype", "fp32", "--rounds", "5", "--warmup", "1"],
        capture_output=True, timeout=120, text=True,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "rounds passed     4 / 4" in out.stdout


def test_zimage_loopback_subprocess_block_propagation():
    """A scale_o fault at block 0 must also poison block 1's MSE."""
    out = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH),
         "--role", "loopback", "--device", "cpu",
         "--dim", "16", "--heads", "2", "--head-dim", "8",
         "--ffn-inter", "32", "--n-layers", "2",
         "--batch", "1", "--seq", "4",
         "--wire-dtype", "fp32", "--rounds", "1", "--warmup", "0",
         "--inject-fault", "scale_o", "--fault-block", "0",
         "--threshold", "1e-10"],
        capture_output=True, timeout=120, text=True,
    )
    assert out.returncode == 0
    assert "rounds passed     0 / 1" in out.stdout
    # Both block 0 (where fault was injected) and block 1 (where verified
    # x_in_for_next diverges from worker's x_in_for_next) must show high MSE
    # on at least one op_kind. We verify by checking the printed mse summary
    # contains TWO different "p95" entries with values > 1e-14.
    high_mse = sum(1 for line in out.stdout.splitlines()
                   if "mse[" in line and "p95" in line
                   and float(line.rsplit()[-1]) > 1e-14)
    assert high_mse >= 2, f"expected ≥2 high-mse ops, got {high_mse}\n{out.stdout}"
