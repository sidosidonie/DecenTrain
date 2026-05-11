"""Performance tests for examples/multi_machine_ffn.py.

Gated by the `perf` marker (not run by default CI). Thresholds are
conservative defaults overridable via env vars; tune per hardware.

Run:
    pytest -q tests/test_multi_machine_ffn_perf.py -m perf
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import socket
import subprocess
import sys
import time
from typing import Optional

import pytest
import torch


pytestmark = pytest.mark.perf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_ffn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mmffn", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmffn"] = mod  # for dataclass annotations under from __future__
    spec.loader.exec_module(mod)
    return mod


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise TimeoutError(f"worker port {port} did not open within {timeout}s")


class _WorkerProc:
    def __init__(self, device: str, **cli):
        self.port = _pick_free_port()
        self.device = device
        self.cli = cli
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        cmd = [sys.executable, str(EXAMPLE_PATH),
               "--role", "worker",
               "--bind", f"127.0.0.1:{self.port}",
               "--device", self.device]
        for k, v in self.cli.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
        self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        _wait_port(self.port)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# Default shape & thresholds: tuned conservatively for "smallish" GPU/CPU.
# Override via env vars when running on faster hardware.
_HIDDEN = _env_int("FFN_PERF_HIDDEN", 1024)
_INTER  = _env_int("FFN_PERF_INTER",  2752)  # ~2.7x hidden
_BATCH  = _env_int("FFN_PERF_BATCH",  1)
_SEQ    = _env_int("FFN_PERF_SEQ",    128)
_ROUNDS = _env_int("FFN_PERF_ROUNDS", 50)
_WARMUP = _env_int("FFN_PERF_WARMUP", 5)

_MIN_ROUND_PER_S = _env_float("FFN_PERF_MIN_ROUND_PER_S", 5.0)
_MAX_P95_MS      = _env_float("FFN_PERF_MAX_P95_MS", 1000.0)
_MIN_WIRE_MBPS   = _env_float("FFN_PERF_MIN_WIRE_MBPS", 50.0)
_MIN_VERIFY_GFLOPS = _env_float("FFN_PERF_MIN_VERIFY_GFLOPS", 0.5)


def _run_perf(inject_fault: str = "none", device: Optional[str] = None):
    m = _load_module()
    device = device or _device()
    with _WorkerProc(device=device, inject_fault=inject_fault) as wp:
        cfg = m.FFNConfig(hidden=_HIDDEN, inter=_INTER, batch=_BATCH, seq=_SEQ,
                          wire_dtype=m.DTYPE_FP16, weight_seed=42)
        coord = m.Coordinator(host="127.0.0.1", port=wp.port, config=cfg,
                              threshold=1e-2)  # fp16 noise floor
        try:
            coord.connect_and_load()
            rounds = coord.run_many(rounds=_ROUNDS)
        finally:
            coord.close()
    measured = rounds[_WARMUP:]
    rates = [m.compute_round_rates(r, cfg, k=m.SLALOM_K) for r in measured]
    return m, cfg, rounds, measured, rates


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _p95(xs):
    import numpy as np
    return float(np.percentile(xs, 95)) if xs else 0.0


def test_steady_state_throughput():
    _m, _cfg, _all, measured, _r = _run_perf()
    mean_e2e = _mean([r.end_to_end_t for r in measured])
    assert mean_e2e > 0
    round_per_s = 1000.0 / mean_e2e
    assert round_per_s >= _MIN_ROUND_PER_S, (
        f"throughput {round_per_s:.2f} round/s < {_MIN_ROUND_PER_S} "
        f"(override with FFN_PERF_MIN_ROUND_PER_S)"
    )


def test_p95_latency_bounded():
    _m, _cfg, _all, measured, _r = _run_perf()
    p95 = _p95([r.end_to_end_t for r in measured])
    assert p95 <= _MAX_P95_MS, (
        f"p95 {p95:.2f}ms > {_MAX_P95_MS}ms "
        f"(override with FFN_PERF_MAX_P95_MS)"
    )


def test_breakdown_sums_close_to_one():
    """Phases are sequential in this example, so sum_pct ≈ 1.0."""
    _m, _cfg, _all, measured, rates = _run_perf()
    mean_sum = _mean([r["sum_pct"] for r in rates])
    assert 0.9 <= mean_sum <= 1.1, (
        f"mean sum_pct {mean_sum:.3f} not in [0.9, 1.1] — "
        f"is the example pipelined unexpectedly?"
    )


def test_wire_throughput_nonzero():
    _m, _cfg, _all, measured, rates = _run_perf()
    mean_mbps = _mean([r["wire_mbps"] for r in rates])
    assert mean_mbps >= _MIN_WIRE_MBPS, (
        f"wire {mean_mbps:.1f} MB/s < {_MIN_WIRE_MBPS} "
        f"(override with FFN_PERF_MIN_WIRE_MBPS)"
    )


def test_gpu_compute_rate_in_valid_range():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _m, _cfg, _all, measured, rates = _run_perf(device="cuda:0")
    mean_gflops = _mean([r["gpu_gflops"] for r in rates])
    assert mean_gflops > 0, "FLOPs computation returned zero"
    # No upper assert (varies wildly by GPU); just lower bound for sanity
    assert mean_gflops > 10.0, f"gpu_gflops {mean_gflops:.1f} suspiciously low"


def test_verify_compute_rate_nonzero():
    _m, _cfg, _all, measured, rates = _run_perf()
    mean_gflops = _mean([r["verify_gflops"] for r in rates])
    assert mean_gflops >= _MIN_VERIFY_GFLOPS, (
        f"verify {mean_gflops:.2f} GFLOPS < {_MIN_VERIFY_GFLOPS} "
        f"(override with FFN_PERF_MIN_VERIFY_GFLOPS)"
    )


def test_warmup_excluded_from_aggregation(tmp_path):
    """Run with high warmup; summary's rounds_measured should reflect it."""
    json_path = tmp_path / "r.json"
    cmd = [sys.executable, str(EXAMPLE_PATH),
           "--role", "loopback",
           "--rounds", "10", "--warmup", "4",
           "--hidden", str(_HIDDEN), "--inter", str(_INTER),
           "--batch", str(_BATCH), "--seq", str(_SEQ),
           "--wire-dtype", "fp16", "--device", _device(),
           "--threshold", "1e-2",
           "--json-report", str(json_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stderr={res.stderr}"
    import json
    data = json.loads(json_path.read_text())
    assert data["summary"]["rounds_total"] == 10
    assert data["summary"]["rounds_warmup"] == 4
    assert data["summary"]["rounds_measured"] == 6


def test_clean_run_passes_verification_at_perf_shape():
    """Sanity: the perf-shape run still verifies cleanly (no flakiness)."""
    _m, _cfg, _all, measured, _r = _run_perf()
    failed = [r for r in measured if not r.ok]
    assert not failed, (
        f"{len(failed)}/{len(measured)} rounds failed verification "
        f"at perf shape. First failure mse={failed[0].mse_w1:.2e}/"
        f"{failed[0].mse_w3:.2e}/{failed[0].mse_w2:.2e}"
    )
