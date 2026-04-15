"""
tests/test_zimage_module_level.py

Module-level unit tests for VerifyLinearModule and VerifyMatmul.

Coverage (per features.md):
- VerifyLinearModule happy path (no bias, with bias)
- VerifyMatmul happy path
- Corruption detection: submit wrong y_gpu -> flush raises RuntimeError
- Corruption detection: submit wrong c_gpu -> flush raises RuntimeError
- Profiling enabled: records are populated after forward + flush
- Profiling disabled: records are empty after forward + flush
- Numeric equivalence: verified output == non-verified output (performance baseline)
- Bonus: direct bias submission exercises freivalds_algorithm_bias path
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.layers import VerifyLinearModule, VerifyMatmul
from verified_diffusers.zimage.runtime import VerifyRuntime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(
    profile_enabled: bool = True,
    fail_on_error: bool = True,
    freivalds_k: int = 6,
    mse_threshold: float = 1e-4,
) -> VerifyRuntime:
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=freivalds_k,
        mse_threshold=mse_threshold,
        max_workers=2,
        profile_enabled=profile_enabled,
        fail_on_error=fail_on_error,
        flush_each_layer=False,
        flush_on_pipeline_end=False,
    )
    return VerifyRuntime(cfg)


def _make_linear_module(
    in_features: int,
    out_features: int,
    bias: bool,
    runtime: VerifyRuntime,
    device: str = "cuda",
):
    linear = nn.Linear(in_features, out_features, bias=bias).to(device)
    linear.eval()
    verify_mod = VerifyLinearModule(linear, runtime, "test.linear")
    verify_mod.to(device)
    return linear, verify_mod


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_verify_linear_no_bias_happy_path():
    """VerifyLinearModule (no bias) produces correct output and passes verification."""
    runtime = _make_runtime()
    in_f, out_f = 32, 64
    linear, verify_mod = _make_linear_module(in_f, out_f, bias=False, runtime=runtime)

    x = torch.randn(4, in_f, device="cuda")
    y_ref = linear(x)
    y_ver = verify_mod(x)

    assert y_ref.shape == y_ver.shape
    assert torch.allclose(y_ref, y_ver, atol=1e-5)

    runtime.flush()
    assert runtime.pending_tasks == 0
    runtime.shutdown()


@torch.no_grad()
def test_verify_linear_with_bias_happy_path():
    """VerifyLinearModule (with bias) produces correct output and passes verification."""
    runtime = _make_runtime()
    in_f, out_f = 32, 64
    linear, verify_mod = _make_linear_module(in_f, out_f, bias=True, runtime=runtime)

    x = torch.randn(4, in_f, device="cuda")
    y_ref = linear(x)
    y_ver = verify_mod(x)

    assert y_ref.shape == y_ver.shape
    assert torch.allclose(y_ref, y_ver, atol=1e-5)

    runtime.flush()
    assert runtime.pending_tasks == 0
    runtime.shutdown()


@torch.no_grad()
def test_verify_matmul_happy_path():
    """VerifyMatmul produces correct output and passes verification."""
    runtime = _make_runtime()
    mm = VerifyMatmul(runtime, "test.matmul")

    a = torch.randn(8, 16, 32, device="cuda")
    b = torch.randn(8, 32, 24, device="cuda")
    c_ref = torch.matmul(a, b)
    c_ver = mm(a, b)

    assert c_ref.shape == c_ver.shape
    assert torch.allclose(c_ref, c_ver, atol=1e-5)

    runtime.flush()
    assert runtime.pending_tasks == 0
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Corruption detection tests
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_verify_linear_corruption_detected():
    """
    Simulated GPU corruption: submit_linear with a random y_gpu (not x @ W^T).
    Freivalds MSE will be large; flush() must raise RuntimeError.
    """
    runtime = _make_runtime(fail_on_error=True, mse_threshold=1e-4)
    in_f, out_f = 32, 64
    linear = nn.Linear(in_f, out_f, bias=False).to("cuda")
    linear.eval()

    x_gpu = torch.randn(4, in_f, device="cuda")
    # Deliberately wrong output — simulates GPU returning garbage
    corrupt_y_gpu = torch.randn(4, out_f, device="cuda")
    weight_t_cpu = linear.weight.detach().t().to("cpu")

    runtime.submit_linear(
        tag="test.corrupt_linear",
        x_gpu=x_gpu,
        y_gpu=corrupt_y_gpu,
        weight_t_cpu=weight_t_cpu,
        bias_cpu=None,
    )

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()

    # shutdown() is safe: _futures=[] and _errors=[] after flush() raised
    runtime.shutdown()


@torch.no_grad()
def test_verify_matmul_corruption_detected():
    """
    Simulated GPU corruption: submit_matmul with a random c_gpu (not a @ b).
    Freivalds MSE will be large; flush() must raise RuntimeError.
    """
    runtime = _make_runtime(fail_on_error=True, mse_threshold=1e-4)

    a = torch.randn(4, 16, 32, device="cuda")
    b = torch.randn(4, 32, 24, device="cuda")
    # Deliberately wrong output
    corrupt_c = torch.randn(4, 16, 24, device="cuda")

    runtime.submit_matmul(
        tag="test.corrupt_matmul",
        a_gpu=a,
        b_gpu=b,
        c_gpu=corrupt_c,
    )

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()

    runtime.shutdown()


# ---------------------------------------------------------------------------
# Profiling on/off tests
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_profiling_enabled_records_populated():
    """With profile_enabled=True, profiler.records must be non-empty after a forward pass."""
    runtime = _make_runtime(profile_enabled=True)
    _, verify_mod = _make_linear_module(32, 64, bias=False, runtime=runtime)

    x = torch.randn(4, 32, device="cuda")
    _ = verify_mod(x)
    runtime.flush()

    records = runtime.profiler.records
    assert len(records) > 0, "Expected profiler records when profile_enabled=True"

    categories = {r.category for r in records}
    assert "verify" in categories, (
        f"Expected 'verify' category in profiler records, got: {categories}"
    )
    assert all(r.ok for r in records), "All records should have ok=True on happy path"

    runtime.shutdown()


@torch.no_grad()
def test_profiling_disabled_records_empty():
    """With profile_enabled=False, profiler.records must remain empty after a forward pass."""
    runtime = _make_runtime(profile_enabled=False)
    _, verify_mod = _make_linear_module(32, 64, bias=False, runtime=runtime)

    x = torch.randn(4, 32, device="cuda")
    _ = verify_mod(x)
    runtime.flush()

    records = runtime.profiler.records
    assert len(records) == 0, (
        f"Expected no profiler records when profile_enabled=False, got {len(records)}"
    )
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Performance comparison baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_verified_linear_matches_unverified_numerics():
    """
    Verified output is numerically identical to the plain nn.Linear output.
    VerifyLinearModule wraps the same object, so the forward path is identical.
    """
    runtime = _make_runtime()
    in_f, out_f = 64, 128

    reference_linear = nn.Linear(in_f, out_f, bias=True).to("cuda")
    reference_linear.eval()
    verified_mod = VerifyLinearModule(reference_linear, runtime, "test.compare")

    x = torch.randn(8, in_f, device="cuda")
    y_ref = reference_linear(x)
    y_ver = verified_mod(x)
    runtime.flush()

    assert torch.allclose(y_ref, y_ver, atol=0.0, rtol=0.0), (
        "Verified and non-verified outputs must be numerically identical"
    )
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Bonus: exercise freivalds_algorithm_bias path via direct submission
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_verify_linear_with_explicit_bias_submission():
    """
    VerifyLinearModule.forward always calls submit_linear(..., bias_cpu=None).
    This test calls submit_linear directly with a real bias_cpu tensor to
    exercise the freivalds_algorithm_bias code path inside VerifyRuntime._verify.
    Submitting the correct y_no_bias must not raise.
    """
    runtime = _make_runtime()
    in_f, out_f = 32, 64
    linear = nn.Linear(in_f, out_f, bias=True).to("cuda")
    linear.eval()

    x_gpu = torch.randn(4, in_f, device="cuda")
    # freivalds_algorithm_bias verifies: x @ W^T + bias == y
    # so y must be the full output INCLUDING bias
    y_with_bias_gpu = F.linear(x_gpu, linear.weight, bias=linear.bias)
    weight_t_cpu = linear.weight.detach().t().to("cpu")
    bias_cpu = linear.bias.detach().to("cpu")

    # This should NOT raise — numbers are consistent
    runtime.submit_linear(
        tag="test.linear_with_bias",
        x_gpu=x_gpu,
        y_gpu=y_with_bias_gpu,
        weight_t_cpu=weight_t_cpu,
        bias_cpu=bias_cpu,
    )

    runtime.flush()
    assert runtime.pending_tasks == 0
    runtime.shutdown()
