"""
tests/test_threat_model.py

End-to-end threat model demonstration.

Simulates a malicious/faulty GPU that corrupts computation at various layers.
Demonstrates that VerifyRuntime detects every type of attack and raises immediately.

Attack scenarios:
  1. Linear layer returns random garbage (GPU completely compromised)
  2. Linear layer adds small noise (subtle data poisoning)
  3. Attention QK matmul returns zeros (GPU skipping computation)
  4. Attention KV matmul is corrupted (wrong attention output)
  5. Softmax output is tampered (element-wise op corruption)
  6. MLP feedforward layer corrupted (MLP path attack)
  7. Multi-layer attack: multiple layers corrupted simultaneously
  8. Selective attack: only corrupt every N-th operation (evasion attempt)

Run:
  pytest tests/test_threat_model.py -v -s
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_core.config import VerifyConfig
from verified_diffusers.zimage.layers import VerifyLinearModule, VerifyMatmul
from verified_core.runtime import VerifyRuntime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(
    fail_on_error: bool = True,
    mse_threshold: float = 1e-4,
    elementwise_mse_threshold: float = 1e-4,
) -> VerifyRuntime:
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=6,
        mse_threshold=mse_threshold,
        elementwise_mse_threshold=elementwise_mse_threshold,
        max_workers=2,
        profile_enabled=True,
        fail_on_error=fail_on_error,
        flush_each_layer=False,
    )
    return VerifyRuntime(cfg)


class CorruptLinearModule(nn.Module):
    """
    A VerifyLinearModule that injects corruption into GPU output.
    Simulates a compromised GPU returning wrong results.
    """

    def __init__(
        self,
        linear: nn.Linear,
        runtime: VerifyRuntime,
        tag: str,
        attack: str = "random",   # "random", "noise", "zero", "scale"
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.linear = linear
        self.runtime = runtime
        self.tag = tag
        self.attack = attack
        self.noise_scale = noise_scale
        self.weight_t_cpu = linear.weight.detach().t().to("cpu")
        self.bias_cpu = linear.bias.detach().to("cpu") if linear.bias is not None else None

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPU computes the real result
        y_real = F.linear(x, self.linear.weight, bias=None)

        # --- ATTACK: corrupt the GPU output ---
        if self.attack == "random":
            # Complete garbage — GPU returns random data
            y_corrupt = torch.randn_like(y_real)
        elif self.attack == "noise":
            # Subtle poisoning — small additive noise
            y_corrupt = y_real + torch.randn_like(y_real) * self.noise_scale
        elif self.attack == "zero":
            # GPU skips computation — returns zeros
            y_corrupt = torch.zeros_like(y_real)
        elif self.attack == "scale":
            # GPU scales output by wrong factor (off-by-one bug)
            y_corrupt = y_real * 2.0
        else:
            y_corrupt = y_real

        # Submit the CORRUPTED output for verification
        # VerifyRuntime will compare against x @ W^T and detect mismatch
        self.runtime.submit_linear(
            tag=self.tag,
            x_gpu=x,
            y_gpu=y_corrupt,
            weight_t_cpu=self.weight_t_cpu,
            bias_cpu=None,
        )

        if self.linear.bias is None:
            return y_corrupt
        return y_corrupt + self.linear.bias


class CorruptMatmul(nn.Module):
    """
    A VerifyMatmul that injects corruption into matmul output.
    Simulates GPU returning wrong Q@K^T or attn@V.
    """

    def __init__(
        self,
        runtime: VerifyRuntime,
        tag: str,
        attack: str = "random",
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.runtime = runtime
        self.tag = tag
        self.attack = attack
        self.noise_scale = noise_scale

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out_real = torch.matmul(a, b)

        if self.attack == "random":
            out_corrupt = torch.randn_like(out_real)
        elif self.attack == "noise":
            out_corrupt = out_real + torch.randn_like(out_real) * self.noise_scale
        elif self.attack == "zero":
            out_corrupt = torch.zeros_like(out_real)
        elif self.attack == "scale":
            out_corrupt = out_real * 2.0
        else:
            out_corrupt = out_real

        self.runtime.submit_matmul(self.tag, a, b, out_corrupt)
        return out_corrupt


# ---------------------------------------------------------------------------
# Attack Scenario 1: Linear returns random garbage
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_linear_random_garbage():
    """
    THREAT: GPU is fully compromised. Linear layer returns random data.
    DETECT: Freivalds MSE is huge → flush() raises RuntimeError.
    """
    runtime = _make_runtime()
    linear = nn.Linear(64, 128, bias=False).to("cuda")

    corrupt = CorruptLinearModule(linear, runtime, "atk.random", attack="random")
    x = torch.randn(4, 64, device="cuda")
    _ = corrupt(x)

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 2: Subtle noise injection (data poisoning)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_linear_subtle_noise():
    """
    THREAT: GPU adds small noise to output (data poisoning / bit flip).
    A noise_scale of 0.1 is large enough for Freivalds to catch.
    DETECT: MSE exceeds threshold → flush() raises.
    """
    runtime = _make_runtime(mse_threshold=1e-5)
    linear = nn.Linear(64, 128, bias=False).to("cuda")

    corrupt = CorruptLinearModule(
        linear, runtime, "atk.noise", attack="noise", noise_scale=0.1
    )
    x = torch.randn(4, 64, device="cuda")
    _ = corrupt(x)

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 3: GPU skips computation (returns zeros)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_linear_skip_computation():
    """
    THREAT: GPU doesn't compute at all — returns zeros.
    This could happen with a lazy/buggy GPU driver.
    DETECT: Freivalds sees ABr ≠ 0 → fail.
    """
    runtime = _make_runtime()
    linear = nn.Linear(64, 128, bias=False).to("cuda")

    corrupt = CorruptLinearModule(linear, runtime, "atk.zero", attack="zero")
    x = torch.randn(4, 64, device="cuda")
    _ = corrupt(x)

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 4: QK matmul corrupted (attention hijack)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_qk_matmul_corrupted():
    """
    THREAT: GPU corrupts Q@K^T in attention, producing wrong attention scores.
    This changes which tokens attend to which — catastrophic for output.
    DETECT: Freivalds on matmul → fail.
    """
    runtime = _make_runtime()
    corrupt_qk = CorruptMatmul(runtime, "atk.qk", attack="random")

    B, H, S, D = 2, 4, 32, 16
    q = torch.randn(B, H, S, D, device="cuda")
    k = torch.randn(B, H, D, S, device="cuda")

    _ = corrupt_qk(q, k)

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 5: KV matmul corrupted (wrong context aggregation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_kv_matmul_corrupted():
    """
    THREAT: GPU corrupts attn_probs @ V, mixing wrong value vectors.
    Model output will be semantically wrong.
    DETECT: Freivalds on matmul → fail.
    """
    runtime = _make_runtime()
    corrupt_kv = CorruptMatmul(runtime, "atk.kv", attack="scale")

    B, H, S, D = 2, 4, 32, 16
    attn_probs = torch.softmax(torch.randn(B, H, S, S, device="cuda"), dim=-1)
    v = torch.randn(B, H, S, D, device="cuda")

    _ = corrupt_kv(attn_probs, v)

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 6: Softmax output tampered
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_softmax_tampered():
    """
    THREAT: GPU returns wrong softmax — e.g., uniform distribution instead
    of peaked. This makes attention ignore the actual scores.
    DETECT: CPU recomputes softmax, MSE is large → fail.
    """
    runtime = _make_runtime(elementwise_mse_threshold=1e-5)

    scores = torch.randn(2, 4, 32, 32, device="cuda")
    # Real softmax
    real_probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    # Corrupt: uniform distribution (GPU ignores scores)
    fake_probs = torch.ones_like(real_probs) / real_probs.shape[-1]

    runtime.submit_elementwise("atk.softmax", scores, fake_probs, "softmax")

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 7: MLP feedforward corruption
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_mlp_feedforward():
    """
    THREAT: One of the 3 MLP projections (gate/up/down) is corrupted.
    Even if attention is fine, wrong MLP output propagates through residual.
    DETECT: Freivalds catches the corrupted linear.
    """
    runtime = _make_runtime()

    # Build a mini MLP: gate, up, down
    gate = nn.Linear(64, 128, bias=False).to("cuda")
    up = nn.Linear(64, 128, bias=False).to("cuda")
    down = nn.Linear(128, 64, bias=False).to("cuda")

    # Only corrupt the down projection — gate and up are honest
    gate_v = VerifyLinearModule(gate, runtime, "mlp.gate")
    up_v = VerifyLinearModule(up, runtime, "mlp.up")
    down_corrupt = CorruptLinearModule(down, runtime, "mlp.down", attack="random")

    x = torch.randn(4, 64, device="cuda")

    # MLP forward: down(silu(gate(x)) * up(x))
    g = gate_v(x)
    u = up_v(x)
    h = F.silu(g) * u
    _ = down_corrupt(h)

    with pytest.raises(RuntimeError, match="Verification failed"):
        runtime.flush()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 8: Multi-layer simultaneous attack
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_multi_layer():
    """
    THREAT: Attacker compromises multiple layers at once.
    - Layer 0 Q-projection: returns random
    - Layer 0 QK matmul: returns zeros
    - Layer 0 down projection: adds noise
    DETECT: All 3 corruptions are detected in a single flush().
    """
    runtime = _make_runtime(fail_on_error=False)

    # Attack 1: corrupt Q projection
    q_linear = nn.Linear(64, 64, bias=False).to("cuda")
    q_corrupt = CorruptLinearModule(q_linear, runtime, "layer0.q", attack="random")
    x = torch.randn(4, 64, device="cuda")
    _ = q_corrupt(x)

    # Attack 2: corrupt QK matmul
    qk_corrupt = CorruptMatmul(runtime, "layer0.qk", attack="zero")
    q = torch.randn(2, 4, 16, 16, device="cuda")
    k = torch.randn(2, 4, 16, 16, device="cuda")
    _ = qk_corrupt(q, k)

    # Attack 3: corrupt down projection with noise
    down = nn.Linear(128, 64, bias=False).to("cuda")
    down_corrupt = CorruptLinearModule(
        down, runtime, "layer0.down", attack="noise", noise_scale=1.0
    )
    h = torch.randn(4, 128, device="cuda")
    _ = down_corrupt(h)

    # flush with fail_on_error=False just collects errors
    runtime.flush()

    # All 3 attacks should be recorded
    errors = runtime._errors
    # Re-read: flush() already consumed _errors. Check profiler instead.
    records = runtime.profiler.records
    failed_records = [r for r in records if r.category == "verify" and not r.ok]
    assert len(failed_records) >= 3, (
        f"Expected at least 3 failed verifications, got {len(failed_records)}.\n"
        f"Failed: {[(r.tag, r.extra) for r in failed_records]}"
    )
    print(f"\n[threat] Detected {len(failed_records)} corrupted operations:")
    for r in failed_records:
        print(f"  FAIL  {r.tag:30s}  {r.op_name:25s}  {r.extra}")

    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 9: Evasion attempt — corrupt only every N-th operation
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_attack_selective_corruption():
    """
    THREAT: Smart attacker only corrupts every 3rd linear to evade sampling.
    With verify_every_n=1 (default), every op is verified → still caught.
    """
    # Use fail_on_error=False so we can inspect all results after flush
    runtime = _make_runtime(fail_on_error=False)

    layers: List[nn.Module] = []
    for i in range(6):
        linear = nn.Linear(64, 64, bias=False).to("cuda")
        if i % 3 == 2:  # Corrupt layers 2 and 5
            layers.append(CorruptLinearModule(
                linear, runtime, f"selective.layer{i}", attack="noise", noise_scale=0.5
            ))
        else:
            layers.append(VerifyLinearModule(linear, runtime, f"selective.layer{i}"))

    x = torch.randn(4, 64, device="cuda")
    for layer in layers:
        x = layer(x)

    runtime.flush()

    # Check that exactly the corrupted layers were caught
    records = runtime.profiler.records
    failed = [r for r in records if r.category == "verify" and not r.ok]
    failed_tags = {r.tag for r in failed}
    assert "selective.layer2" in failed_tags, "Layer 2 corruption not detected"
    assert "selective.layer5" in failed_tags, "Layer 5 corruption not detected"

    # Honest layers should pass
    passed = [r for r in records if r.category == "verify" and r.ok]
    passed_tags = {r.tag for r in passed}
    for honest_idx in [0, 1, 3, 4]:
        assert f"selective.layer{honest_idx}" in passed_tags, (
            f"Honest layer {honest_idx} should have passed verification"
        )

    print(f"\n[threat] Selective attack: {len(failed)} corrupted detected, "
          f"{len(passed)} honest passed")
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Attack Scenario 10: Verify honest pipeline works (control group)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_honest_pipeline_passes():
    """
    CONTROL: No corruption. All verified operations should pass.
    This confirms we don't have false positives.
    """
    runtime = _make_runtime()

    # Build a mini attention + MLP pipeline
    q_proj = VerifyLinearModule(nn.Linear(64, 64, bias=False).to("cuda"), runtime, "honest.q")
    k_proj = VerifyLinearModule(nn.Linear(64, 64, bias=False).to("cuda"), runtime, "honest.k")
    v_proj = VerifyLinearModule(nn.Linear(64, 64, bias=False).to("cuda"), runtime, "honest.v")
    o_proj = VerifyLinearModule(nn.Linear(64, 64, bias=False).to("cuda"), runtime, "honest.o")
    qk_mm = VerifyMatmul(runtime, "honest.qk")
    kv_mm = VerifyMatmul(runtime, "honest.kv")
    ff_w1 = VerifyLinearModule(nn.Linear(64, 128, bias=False).to("cuda"), runtime, "honest.w1")
    ff_w2 = VerifyLinearModule(nn.Linear(128, 64, bias=False).to("cuda"), runtime, "honest.w2")

    x = torch.randn(2, 16, 64, device="cuda")

    # Attention
    q = q_proj(x).unflatten(-1, (4, 16)).permute(0, 2, 1, 3)
    k = k_proj(x).unflatten(-1, (4, 16)).permute(0, 2, 1, 3)
    v = v_proj(x).unflatten(-1, (4, 16)).permute(0, 2, 1, 3)

    scores = qk_mm(q, k.transpose(2, 3)) * (16 ** -0.5)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
    runtime.submit_elementwise("honest.softmax", scores, probs, "softmax")
    attn_out = kv_mm(probs, v)
    attn_out = attn_out.permute(0, 2, 1, 3).flatten(2, 3)
    attn_out = o_proj(attn_out)

    # MLP
    h = ff_w1(attn_out)
    h = F.silu(h)
    out = ff_w2(h)

    # Should NOT raise — everything is honest
    runtime.flush()

    records = runtime.profiler.records
    verify_records = [r for r in records if r.category == "verify"]
    failed = [r for r in verify_records if not r.ok]
    assert len(failed) == 0, (
        f"False positives! {len(failed)} honest ops flagged:\n"
        + "\n".join(f"  {r.tag}: {r.extra}" for r in failed)
    )
    print(f"\n[threat] Honest pipeline: {len(verify_records)} ops verified, 0 false positives")
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Summary test: print threat model coverage
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_threat_model_summary():
    """
    Summary: demonstrates the full threat model coverage.
    Not a real test — just prints the attack matrix.
    """
    attacks = [
        ("Random garbage",    "Linear returns random data",          "Freivalds MSE >> threshold"),
        ("Subtle noise",      "Small additive noise (poisoning)",    "Freivalds MSE > threshold"),
        ("Skip computation",  "GPU returns zeros",                   "Freivalds detects ABr ≠ 0"),
        ("QK matmul corrupt", "Wrong attention scores",              "Freivalds on batched matmul"),
        ("KV matmul corrupt", "Wrong value aggregation",             "Freivalds on batched matmul"),
        ("Softmax tampered",  "Uniform instead of peaked",           "CPU recomputes, MSE check"),
        ("MLP corruption",    "One of gate/up/down is wrong",        "Freivalds per projection"),
        ("Multi-layer",       "Multiple layers attacked at once",    "All detected independently"),
        ("Selective attack",  "Only every N-th op corrupted",        "verify_every_n=1 catches all"),
        ("Honest (control)",  "No corruption",                       "Zero false positives"),
    ]

    print("\n" + "=" * 85)
    print("THREAT MODEL COVERAGE")
    print("=" * 85)
    print(f"{'Attack':20s} {'Description':38s} {'Detection Method':28s}")
    print("-" * 85)
    for name, desc, method in attacks:
        print(f"{name:20s} {desc:38s} {method:28s}")
    print("=" * 85)
