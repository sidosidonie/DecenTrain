"""Functional tests for examples/multi_machine_ffn.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Optional


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_ffn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mmffn", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so that `from __future__ import annotations`
    # dataclasses can resolve string annotations against the module's own ns
    # (dataclass internals look up cls.__module__ in sys.modules).
    sys.modules["mmffn"] = mod
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


def test_slalom_verify_safe_returns_inf_on_nan_y():
    """NaN-poisoned y must trip the safe wrapper (not silently pass)."""
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42,
                                dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    y[0, 0, 0] = float("nan")
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    mse = m.slalom_verify_safe(x, y, s, s_tilde)
    assert mse == float("inf")


def test_slalom_verify_safe_returns_inf_on_inf_y():
    """Inf in y must also trip the safe wrapper."""
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42,
                                dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    y[0, 0, 0] = float("inf")
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    mse = m.slalom_verify_safe(x, y, s, s_tilde)
    assert mse == float("inf")


def test_slalom_verify_safe_matches_slalom_verify_when_finite():
    """For clean y, safe wrapper returns identical mse to base function."""
    m = _load_module()
    torch.manual_seed(0)
    w1, _, _ = m.make_weights(hidden=8, inter=16, seed=42,
                                dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, 8)
    y = w1(x)
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(w1.weight, s)
    base = m.slalom_verify(x, y, s, s_tilde)
    safe = m.slalom_verify_safe(x, y, s, s_tilde)
    assert base == safe


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


def _run_one_round_collect_acts(wp, *, hidden, inter, input_seed=99,
                                 batch=1, seq=4):
    m = _load_module()
    with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
        m.send_msg(sock, m.MSG_LOAD_REQ,
                   m.pack_load_req(hidden=hidden, inter=inter,
                                    weight_seed=42, dtype_id=m.DTYPE_FP32))
        m.recv_msg(sock)
        m.send_msg(sock, m.MSG_FORWARD_REQ,
                   m.pack_forward_req(request_id=1, input_seed=input_seed,
                                       batch=batch, seq=seq))
        acts = {}
        while len(acts) < 3:
            mt, body = m.recv_msg(sock)
            if mt == m.MSG_ACTIVATION:
                d = m.unpack_activation(body)
                acts[d["op_tag"]] = d["tensor"]
        m.recv_msg(sock)  # FORWARD_DONE
        m.send_msg(sock, m.MSG_CLOSE, b"")
    return acts


def test_fault_flip_y1_negates_first_output():
    m = _load_module()
    with _WorkerProc(inject_fault="none") as clean, \
         _WorkerProc(inject_fault="flip_y1") as bad:
        clean_acts = _run_one_round_collect_acts(clean, hidden=8, inter=16)
        bad_acts = _run_one_round_collect_acts(bad, hidden=8, inter=16)
        assert torch.allclose(bad_acts[m.OP_W1], -clean_acts[m.OP_W1])
        # w3 unaffected
        assert torch.allclose(bad_acts[m.OP_W3], clean_acts[m.OP_W3])


def test_fault_scale_y2_scales_third_output():
    m = _load_module()
    with _WorkerProc(inject_fault="none") as clean, \
         _WorkerProc(inject_fault="scale_y2") as bad:
        clean_acts = _run_one_round_collect_acts(clean, hidden=8, inter=16)
        bad_acts = _run_one_round_collect_acts(bad, hidden=8, inter=16)
        # First catch the no-op regression: bad must actually differ from clean.
        assert not torch.allclose(bad_acts[m.OP_W2], clean_acts[m.OP_W2])
        # Then verify the scale factor.
        assert torch.allclose(bad_acts[m.OP_W2], clean_acts[m.OP_W2] * 1.01,
                              atol=1e-4)


def test_fault_drop_silu_changes_y2_via_chain():
    m = _load_module()
    with _WorkerProc(inject_fault="none") as clean, \
         _WorkerProc(inject_fault="drop_silu") as bad:
        clean_acts = _run_one_round_collect_acts(clean, hidden=8, inter=16)
        bad_acts = _run_one_round_collect_acts(bad, hidden=8, inter=16)
        # y1 and y3 same as clean (linear projections unaffected)
        assert torch.allclose(bad_acts[m.OP_W1], clean_acts[m.OP_W1])
        assert torch.allclose(bad_acts[m.OP_W3], clean_acts[m.OP_W3])
        # but y2 differs because gated was computed without SiLU
        assert not torch.allclose(bad_acts[m.OP_W2], clean_acts[m.OP_W2])


def test_coordinator_connect_loads_weights_and_builds_slalom():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            # Coordinator owns CPU-side weights and SLALOM state
            assert coord.w1.weight.shape == (16, 8)
            assert coord.s_w1.shape == (16, m.SLALOM_K)
            assert coord.s_tilde_w1.shape == (8, m.SLALOM_K)
            assert coord.s_w2.shape == (8, m.SLALOM_K)
            assert coord.s_tilde_w2.shape == (16, m.SLALOM_K)
        finally:
            coord.close()


def test_coordinator_run_round_clean_passes_verification():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            rm = coord.run_round(request_id=1, input_seed=99)
            assert rm.ok is True
            assert rm.mse_w1 < 1e-3
            assert rm.mse_w3 < 1e-3
            assert rm.mse_w2 < 1e-3
        finally:
            coord.close()


def test_coordinator_run_round_with_flip_y1_caught():
    m = _load_module()
    with _WorkerProc(inject_fault="flip_y1") as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            rm = coord.run_round(request_id=1, input_seed=99)
            assert rm.ok is False
            assert rm.mse_w1 > 1e-3
        finally:
            coord.close()


def test_coordinator_run_many_rounds_collects_metrics():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            ms = coord.run_many(rounds=5)
            assert len(ms) == 5
            assert all(r.ok for r in ms)
            assert all(r.bytes_recv > 0 for r in ms)
        finally:
            coord.close()


def test_bytes_recv_matches_predicted_within_one_percent():
    m = _load_module()
    with _WorkerProc() as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-3)
        try:
            coord.connect_and_load()
            ms = coord.run_many(rounds=3)
            for r in ms:
                rel = abs(r.bytes_recv - r.bytes_recv_predicted) / r.bytes_recv_predicted
                assert rel < 0.01, f"round {r.request_id}: predicted {r.bytes_recv_predicted} got {r.bytes_recv}"
        finally:
            coord.close()


def test_compute_rates_for_round():
    m = _load_module()
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32)
    r = m.RoundMetrics(request_id=0,
                       gpu_forward_t=2.0, wire_recv_t=1.0, cpu_verify_t=4.0,
                       end_to_end_t=8.0, bytes_recv=49000,
                       bytes_recv_predicted=49000)
    rates = m.compute_round_rates(r, cfg, k=m.SLALOM_K)
    assert rates["wire_mbps"] > 0
    assert rates["gpu_gflops"] > 0
    assert rates["verify_gflops"] > 0
    # Phase breakdown
    assert 0 <= rates["gpu_pct"] <= 1
    assert 0 <= rates["wire_pct"] <= 1
    assert 0 <= rates["verify_pct"] <= 1


def test_format_summary_includes_required_sections():
    m = _load_module()
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32, weight_seed=42)
    rounds_metrics = [
        m.RoundMetrics(request_id=i, gpu_forward_t=2.0, wire_recv_t=1.0,
                       cpu_verify_t=4.0, end_to_end_t=8.0,
                       bytes_recv=49000, bytes_recv_predicted=49000,
                       mse_w1=1e-6, mse_w3=1e-6, mse_w2=1e-6, ok=True)
        for i in range(10)
    ]
    text = m.format_summary(rounds_metrics, cfg, warmup=2, k=m.SLALOM_K)
    for needle in [
        "Multi-Machine FFN Example",
        "Phase timings",
        "Phase breakdown",
        "Rate breakdown",
        "Verification:",
        "Wire bytes",
    ]:
        assert needle in text, f"missing: {needle}"
    assert "rounds passed     8 / 8" in text  # 10 - 2 warmup


def test_write_json_report_schema(tmp_path):
    m = _load_module()
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32, weight_seed=42)
    rounds_metrics = [
        m.RoundMetrics(request_id=i, gpu_forward_t=2.0, wire_recv_t=1.0,
                       cpu_verify_t=4.0, end_to_end_t=8.0,
                       bytes_recv=49000, bytes_recv_predicted=49000,
                       mse_w1=1e-6, mse_w3=1e-6, mse_w2=1e-6, ok=True)
        for i in range(3)
    ]
    out = tmp_path / "report.json"
    m.write_json_report(out, rounds_metrics, cfg, warmup=1, k=m.SLALOM_K)
    import json as _json
    data = _json.loads(out.read_text())
    assert "config" in data and "per_round" in data and "summary" in data
    assert len(data["per_round"]) == 3
    assert "mean_round_per_s" in data["summary"]
    assert "p95_end_to_end_ms" in data["summary"]
    assert "mean_wire_mbps" in data["summary"]
    assert "rounds_passed" in data["summary"]


def test_loopback_role_runs_end_to_end(tmp_path):
    """`--role loopback` spawns its own worker and runs N rounds clean."""
    json_path = tmp_path / "r.json"
    cmd = [_sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "5", "--warmup", "1",
           "--hidden", "8", "--inter", "16",
           "--batch", "1", "--seq", "4",
           "--wire-dtype", "fp32",
           "--device", "cpu",
           "--json-report", str(json_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "Multi-Machine FFN Example" in res.stdout
    import json as _json
    data = _json.loads(json_path.read_text())
    assert data["summary"]["rounds_passed"] == 4  # 5 - 1 warmup


def test_close_message_shuts_worker_within_two_seconds():
    m = _load_module()
    with _WorkerProc() as wp:
        with _socket.create_connection(("127.0.0.1", wp.port)) as sock:
            m.send_msg(sock, m.MSG_LOAD_REQ,
                       m.pack_load_req(hidden=8, inter=16,
                                        weight_seed=42, dtype_id=m.DTYPE_FP32))
            m.recv_msg(sock)
            m.send_msg(sock, m.MSG_CLOSE, b"")
        rc = wp.proc.wait(timeout=2)
        assert rc == 0


def test_wire_dtype_fp16_runs_clean(tmp_path):
    cmd = [_sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "3", "--warmup", "0",
           "--hidden", "16", "--inter", "32",
           "--batch", "1", "--seq", "4",
           "--wire-dtype", "fp16",
           "--device", "cpu",
           "--threshold", "1e-2"]  # fp16 round-trip is noisier
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "rounds passed     3 / 3" in res.stdout


def test_inject_scale_y2_caught_through_coordinator():
    # At small dims (inter=16) y2*1.01 produces mse_w2 ~3e-12..7e-11,
    # well above the clean baseline (~1e-20) but far below the 1e-3
    # threshold used in other tests. We calibrate the threshold to the
    # test setup (mirrors Task 3's 1e-7 unit-test calibration). Production
    # thresholds at inter=11008 are higher because the same forgery
    # scales linearly with output dim.
    m = _load_module()
    threshold = 1e-12
    with _WorkerProc(inject_fault="scale_y2") as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=threshold)
        try:
            coord.connect_and_load()
            results = coord.run_many(rounds=3)
        finally:
            coord.close()
        assert all(not r.ok for r in results)
        assert all(r.mse_w2 > threshold for r in results)
        # w1, w3 should still pass (only y2 was tampered with)
        assert all(r.mse_w1 < threshold and r.mse_w3 < threshold
                   for r in results)


def test_coordinator_rejects_wrong_shape_activation():
    """If worker sends ACTIVATION with wrong shape, coord raises WireProtocolError."""
    import pytest
    m = _load_module()
    # Build a coordinator (we'll swap its socket for a socketpair so no
    # real worker is needed for this path).
    cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                      wire_dtype=m.DTYPE_FP32, weight_seed=42)
    coord = m.Coordinator(host="127.0.0.1", port=1, config=cfg,
                          threshold=1e-3)
    a, b = _socket.socketpair()
    try:
        # Replace coord's sock with one end of the pair. Coordinator will
        # write FORWARD_REQ into `a` (buffered by the kernel — nobody reads
        # it) and then try to read a response. We pre-write a wrong-shape
        # ACTIVATION into the other side so coord reads that.
        coord.sock = a
        wrong_tensor = torch.zeros(1, 4, 99)  # wrong inter dim (99 != 16)
        body = m.pack_activation(request_id=1, op_tag=m.OP_W1,
                                  tensor=wrong_tensor,
                                  wire_dtype_id=m.DTYPE_FP32)
        m.send_msg(b, m.MSG_ACTIVATION, body)
        with pytest.raises(m.WireProtocolError, match="expected shape"):
            coord.run_round(request_id=1, input_seed=99)
    finally:
        try:
            a.close()
        except OSError:
            pass
        try:
            b.close()
        except OSError:
            pass
        # Avoid double-close on coord.sock (already closed via `a` above).
        coord.sock = None
        coord.pool.shutdown(wait=False)


def test_inject_drop_silu_caught_via_chain():
    # Worker's broken SiLU is caught by the third SLALOM check because
    # coord computes SiLU on CPU from its own y1, y3, so the gated input
    # to w2's check differs from what the worker actually used. drop_silu
    # at inter=16 yields mse_w2 ~3e-8..5e-7 — calibrated threshold 1e-12
    # catches it cleanly while clean rounds (mse ~1e-20) stay below.
    m = _load_module()
    threshold = 1e-12
    with _WorkerProc(inject_fault="drop_silu") as wp:
        cfg = m.FFNConfig(hidden=8, inter=16, batch=1, seq=4,
                          wire_dtype=m.DTYPE_FP32, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=threshold)
        try:
            coord.connect_and_load()
            results = coord.run_many(rounds=3)
        finally:
            coord.close()
        assert all(not r.ok for r in results)
        assert all(r.mse_w2 > threshold for r in results)
        assert all(r.mse_w1 < threshold and r.mse_w3 < threshold
                   for r in results)


def test_pipeline_mode_verifies_and_overlaps(tmp_path):
    """--pipeline keeps verification correct AND overlaps verify with recv,
    so the per-round sum (gpu + wire + verify) exceeds end-to-end time —
    i.e. mean_sum_pct > 1.0. The non-pipelined run is ~1.0."""
    json_path = tmp_path / "pipe.json"
    cmd = [_sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback", "--pipeline",
           "--rounds", "12", "--warmup", "2",
           "--hidden", "256", "--inter", "704",
           "--batch", "1", "--seq", "32",
           "--wire-dtype", "fp16", "--device", "cpu",
           "--json-report", str(json_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "Pipeline: on" in res.stdout
    import json as _json
    data = _json.loads(json_path.read_text())
    assert data["summary"]["rounds_passed"] == 10  # 12 - 2 warmup
    # Overlap signature: the three phases together exceed wall-clock e2e.
    assert data["summary"]["mean_sum_pct"] > 1.0, data["summary"]


def test_non_pipeline_mode_phases_are_sequential(tmp_path):
    """Without --pipeline, mean_sum_pct ≈ 1.0 (phases run back-to-back)."""
    json_path = tmp_path / "seq.json"
    cmd = [_sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "12", "--warmup", "2",
           "--hidden", "256", "--inter", "704",
           "--batch", "1", "--seq", "32",
           "--wire-dtype", "fp16", "--device", "cpu",
           "--json-report", str(json_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "Pipeline: off" in res.stdout
    import json as _json
    data = _json.loads(json_path.read_text())
    assert data["summary"]["rounds_passed"] == 10
    # Sequential: allow generous tolerance for timing jitter.
    assert 0.85 <= data["summary"]["mean_sum_pct"] <= 1.15, data["summary"]
