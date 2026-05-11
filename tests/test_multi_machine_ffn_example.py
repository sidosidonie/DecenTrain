"""Functional tests for examples/multi_machine_ffn.py."""
from __future__ import annotations

import importlib.util
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_ffn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mmffn", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_and_exposes_constants():
    m = _load_module()
    assert m.MSG_LOAD_REQ == 1
    assert m.MSG_LOAD_ACK == 2
    assert m.MSG_FORWARD_REQ == 3
    assert m.MSG_ACTIVATION == 4
    assert m.MSG_FORWARD_DONE == 5
    assert m.MSG_CLOSE == 6
    assert m.OP_W1 == 1
    assert m.OP_W3 == 2
    assert m.OP_W2 == 3
    assert m.DTYPE_FP32 == 1
    assert m.DTYPE_FP16 == 2


import torch


def test_make_weights_deterministic():
    m = _load_module()
    w1_a, w2_a, w3_a = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    w1_b, w2_b, w3_b = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    assert torch.equal(w1_a.weight, w1_b.weight)
    assert torch.equal(w2_a.weight, w2_b.weight)
    assert torch.equal(w3_a.weight, w3_b.weight)


def test_make_weights_shapes():
    m = _load_module()
    w1, w2, w3 = m.make_weights(hidden=8, inter=16, seed=0, dtype=torch.float32, device="cpu")
    assert w1.weight.shape == (16, 8)
    assert w2.weight.shape == (8, 16)
    assert w3.weight.shape == (16, 8)
    assert w1.bias is None and w2.bias is None and w3.bias is None


def test_make_weights_cross_device_dtype_consistency():
    """CPU fp32 and (would-be) GPU fp16 produce same values up to dtype cast."""
    m = _load_module()
    w1_cpu, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    w1_half, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float16, device="cpu")
    # fp32 -> fp16 -> fp32 should match what we got from direct fp16
    assert torch.allclose(w1_cpu.weight.to(torch.float16).to(torch.float32),
                          w1_half.weight.to(torch.float32))


def test_slalom_verify_passes_for_correct_matmul():
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    mse = m.slalom_verify(x, y, s, s_tilde)
    assert mse < 1e-6, f"expected ~0 mse, got {mse}"


def test_slalom_verify_catches_corrupted_y():
    """SLALOM detects a 1% scale forgery. Threshold is calibrated for the
    small-dim test setup (inter=16, weight std=0.02) — the MSE of a
    correct pair is ~1e-15, the MSE of a 1% scaled forgery is ~6e-6, so
    1e-7 is comfortably between them. Production thresholds at full FFN
    dims are higher because inter=11008 amplifies the same forgery
    proportionally."""
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    y_bad = y * 1.01
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    mse = m.slalom_verify(x, y_bad, s, s_tilde)
    assert mse > 1e-7, f"expected detectable mse, got {mse}"


def test_precompute_s_tilde_shape():
    m = _load_module()
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42, dtype=torch.float32, device="cpu")
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    assert s_tilde.shape == (8, 10)  # (in, k)


def test_make_s_deterministic_with_seed():
    m = _load_module()
    a = m.make_s(out_dim=16, k=10, seed=7)
    b = m.make_s(out_dim=16, k=10, seed=7)
    assert torch.equal(a, b)


import socket as _socket


def test_send_recv_msg_roundtrip():
    m = _load_module()
    a, b = _socket.socketpair()
    try:
        m.send_msg(a, m.MSG_LOAD_REQ, b"hello")
        msg_type, body = m.recv_msg(b)
        assert msg_type == m.MSG_LOAD_REQ
        assert body == b"hello"
    finally:
        a.close()
        b.close()


def test_send_recv_msg_empty_body():
    m = _load_module()
    a, b = _socket.socketpair()
    try:
        m.send_msg(a, m.MSG_CLOSE, b"")
        msg_type, body = m.recv_msg(b)
        assert msg_type == m.MSG_CLOSE
        assert body == b""
    finally:
        a.close()
        b.close()


def test_recv_exactly_raises_on_eof():
    import pytest
    m = _load_module()
    a, b = _socket.socketpair()
    a.close()  # immediately
    try:
        with pytest.raises(ConnectionError):
            m.recv_exactly(b, 10)
    finally:
        b.close()
