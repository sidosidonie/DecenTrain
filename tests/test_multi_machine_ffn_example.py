"""Functional tests for examples/multi_machine_ffn.py."""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Optional


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
import torch.nn.functional as F


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


def test_pack_unpack_load_req():
    m = _load_module()
    body = m.pack_load_req(hidden=4096, inter=11008, weight_seed=0xC0FFEE, dtype_id=m.DTYPE_FP16)
    out = m.unpack_load_req(body)
    assert out == {"hidden": 4096, "inter": 11008, "weight_seed": 0xC0FFEE, "dtype_id": m.DTYPE_FP16}


def test_pack_unpack_forward_req():
    m = _load_module()
    body = m.pack_forward_req(request_id=7, input_seed=99, batch=1, seq=512)
    out = m.unpack_forward_req(body)
    assert out == {"request_id": 7, "input_seed": 99, "batch": 1, "seq": 512}


def test_pack_unpack_activation_roundtrip():
    m = _load_module()
    t = torch.randn(2, 4, 8, dtype=torch.float16)
    body = m.pack_activation(request_id=3, op_tag=m.OP_W1, tensor=t, wire_dtype_id=m.DTYPE_FP16)
    out = m.unpack_activation(body)
    assert out["request_id"] == 3
    assert out["op_tag"] == m.OP_W1
    assert out["tensor"].shape == (2, 4, 8)
    assert out["tensor"].dtype == torch.float16
    assert torch.allclose(out["tensor"], t)


def test_pack_unpack_forward_done():
    m = _load_module()
    body = m.pack_forward_done(request_id=5, gpu_forward_t_ms=12.345)
    out = m.unpack_forward_done(body)
    assert out["request_id"] == 5
    assert abs(out["gpu_forward_t_ms"] - 12.345) < 1e-9


import subprocess
import time as _time
import sys as _sys


def _pick_free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> None:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            with _socket.create_connection((host, port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            _time.sleep(0.05)
    raise TimeoutError(f"worker port {port} did not open within {timeout}s")


class _WorkerProc:
    """Context manager that spawns a worker subprocess on a free port."""

    def __init__(self, **cli):
        self.cli = cli
        self.port = _pick_free_port()
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        cmd = [_sys.executable, str(EXAMPLE_PATH),
               "--role", "worker",
               "--bind", f"127.0.0.1:{self.port}",
               "--device", "cpu"]  # tests default to CPU for portability
        for k, v in self.cli.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            _wait_port(self.port)
        except TimeoutError:
            self.proc.terminate()
            out, err = self.proc.communicate(timeout=2)
            raise RuntimeError(
                f"worker did not start. stdout={out!r} stderr={err!r}"
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def test_worker_handles_load_and_close():
    """Bring up worker, send LOAD_REQ, get LOAD_ACK, send CLOSE, worker exits."""
    m = _load_module()
    with _WorkerProc() as wp:
        with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
            m.send_msg(sock, m.MSG_LOAD_REQ,
                       m.pack_load_req(hidden=8, inter=16,
                                        weight_seed=42, dtype_id=m.DTYPE_FP32))
            mt, body = m.recv_msg(sock)
            assert mt == m.MSG_LOAD_ACK
            assert m.unpack_load_ack(body)["status"] == 0
            m.send_msg(sock, m.MSG_CLOSE, b"")
        # worker should exit cleanly within 3s
        rc = wp.proc.wait(timeout=3)
        assert rc == 0, f"worker exited with {rc}"


def _reproduce_input(seed: int, batch: int, seq: int, hidden: int,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, seq, hidden, dtype=torch.float32, generator=gen).to(dtype)


def test_worker_forward_produces_correct_outputs():
    m = _load_module()
    with _WorkerProc() as wp:
        with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
            m.send_msg(sock, m.MSG_LOAD_REQ,
                       m.pack_load_req(hidden=8, inter=16,
                                        weight_seed=42, dtype_id=m.DTYPE_FP32))
            mt, _ = m.recv_msg(sock)
            assert mt == m.MSG_LOAD_ACK

            m.send_msg(sock, m.MSG_FORWARD_REQ,
                       m.pack_forward_req(request_id=1, input_seed=99,
                                           batch=1, seq=4))
            # Expect 3 ACTIVATION + 1 FORWARD_DONE
            received = []
            for _ in range(4):
                mt, body = m.recv_msg(sock)
                received.append((mt, body))
            m.send_msg(sock, m.MSG_CLOSE, b"")

        acts = [b for (t, b) in received if t == m.MSG_ACTIVATION]
        dones = [b for (t, b) in received if t == m.MSG_FORWARD_DONE]
        assert len(acts) == 3 and len(dones) == 1

        # Reproduce expected outputs from same seed + same weights on CPU
        w1, w2, w3 = m.make_weights(hidden=8, inter=16, seed=42,
                                     dtype=torch.float32, device="cpu")
        x = _reproduce_input(seed=99, batch=1, seq=4, hidden=8)
        y1_expected = w1(x)
        y3_expected = w3(x)
        y2_expected = w2(F.silu(y1_expected) * y3_expected)

        by_tag = {}
        for body in acts:
            d = m.unpack_activation(body)
            by_tag[d["op_tag"]] = d["tensor"]
        assert torch.allclose(by_tag[m.OP_W1], y1_expected, atol=1e-4)
        assert torch.allclose(by_tag[m.OP_W3], y3_expected, atol=1e-4)
        assert torch.allclose(by_tag[m.OP_W2], y2_expected, atol=1e-4)

        done = m.unpack_forward_done(dones[0])
        assert done["request_id"] == 1
        assert done["gpu_forward_t_ms"] > 0
