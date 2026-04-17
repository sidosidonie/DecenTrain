"""
tests/test_e2e_llm.py

End-to-end LLM verification tests using Qwen2.5-0.5B-Instruct (small, fast).
Demonstrates that verified inference produces identical results to unverified,
and that corruption is detected.

Requires:
  - CUDA GPU
  - transformers, torch installed
  - Model will be auto-downloaded (~1GB for 0.5B)

Run:
  RUN_E2E_LLM=1 pytest tests/test_e2e_llm.py -v -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_core.config import VerifyConfig
from verified_core.runtime import VerifyRuntime
from verified_llm.llm_model import create_llm_model

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        os.getenv("RUN_E2E_LLM", "0") != "1",
        reason="Set RUN_E2E_LLM=1 to run (requires GPU + model download)",
    ),
]

# Use a small model for fast CI-friendly tests
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float32  # float32 for exact comparison
PROMPT = "The capital of France is"


def _tokenize(model, text: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    return tokenizer, inputs


@torch.no_grad()
def test_e2e_llm_logit_equivalence():
    """
    Verified model must produce identical logits to the original model.
    Same weights, same input, same dtype → bitwise identical forward pass.
    """
    # --- Original model ---
    model_origin = create_llm_model(
        MODEL_ID, verify=False, dtype=DTYPE, device="cuda",
        trust_remote_code=True,
    )
    model_origin.eval()

    tokenizer, inputs = _tokenize(model_origin, PROMPT)
    logits_origin = model_origin(**inputs).logits

    # Free original model to save VRAM
    del model_origin
    torch.cuda.empty_cache()

    # --- Verified model ---
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        elementwise_mse_threshold=1e-3,
        fail_on_error=True,
        profile_enabled=True,
    )
    model_verify = create_llm_model(
        MODEL_ID, verify=True, config=cfg, dtype=DTYPE, device="cuda",
        trust_remote_code=True,
    )
    model_verify.eval()

    inputs_v = tokenizer(PROMPT, return_tensors="pt").to(model_verify.device)
    logits_verify = model_verify(**inputs_v).logits

    runtime = model_verify._verify_runtime
    runtime.flush()

    # --- Compare ---
    assert logits_origin.shape == logits_verify.shape, (
        f"Shape mismatch: {logits_origin.shape} vs {logits_verify.shape}"
    )

    mse = torch.nn.functional.mse_loss(logits_origin, logits_verify).item()
    max_diff = (logits_origin - logits_verify).abs().max().item()
    is_close = torch.allclose(logits_origin, logits_verify, atol=1e-4, rtol=1e-4)

    print(f"\n[e2e-llm] Logit MSE:      {mse:.8f}")
    print(f"[e2e-llm] Logit max diff: {max_diff:.8f}")
    print(f"[e2e-llm] allclose:       {is_close}")

    # Check profiler — all checks should pass
    records = runtime.profiler.records if runtime.profiler else []
    n_total = len(records)
    n_fail = sum(1 for r in records if not r.ok)
    print(f"[e2e-llm] Verify checks: {n_total} total, {n_fail} failures")

    runtime.shutdown()
    del model_verify
    torch.cuda.empty_cache()

    assert n_fail == 0, f"{n_fail}/{n_total} verification checks failed"
    assert is_close, f"Logits differ: MSE={mse:.8f}, max_diff={max_diff:.8f}"


@torch.no_grad()
def test_e2e_llm_greedy_generation():
    """
    Greedy decoding must produce the same tokens with and without verification.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    max_new_tokens = 20

    # --- Original ---
    model_origin = create_llm_model(
        MODEL_ID, verify=False, dtype=DTYPE, device="cuda",
        trust_remote_code=True,
    )
    model_origin.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    out_origin = model_origin.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    text_origin = tokenizer.decode(out_origin[0], skip_special_tokens=True)

    del model_origin
    torch.cuda.empty_cache()

    # --- Verified ---
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        elementwise_mse_threshold=1e-3,
        fail_on_error=True,
        profile_enabled=True,
    )
    model_verify = create_llm_model(
        MODEL_ID, verify=True, config=cfg, dtype=DTYPE, device="cuda",
        trust_remote_code=True,
    )
    model_verify.eval()

    inputs_v = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    out_verify = model_verify.generate(
        **inputs_v, max_new_tokens=max_new_tokens, do_sample=False,
    )
    text_verify = tokenizer.decode(out_verify[0], skip_special_tokens=True)

    runtime = model_verify._verify_runtime
    runtime.flush()

    print(f"\n[e2e-llm] Origin text:   {text_origin!r}")
    print(f"[e2e-llm] Verified text: {text_verify!r}")

    records = runtime.profiler.records if runtime.profiler else []
    n_total = len(records)
    n_fail = sum(1 for r in records if not r.ok)
    print(f"[e2e-llm] Verify checks: {n_total} total, {n_fail} failures")

    runtime.shutdown()
    del model_verify
    torch.cuda.empty_cache()

    assert n_fail == 0, f"{n_fail}/{n_total} verification checks failed"
    assert torch.equal(out_origin.cpu(), out_verify.cpu()), (
        f"Token mismatch!\nOrigin:   {text_origin!r}\nVerified: {text_verify!r}"
    )


@torch.no_grad()
def test_e2e_llm_corruption_detected():
    """
    Inject noise into a verified model's linear layer weights on GPU,
    simulating a malicious GPU returning wrong results.
    Verification must detect the corruption.
    """
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=8,
        mse_threshold=1e-5,
        elementwise_mse_threshold=1e-4,
        fail_on_error=False,  # collect all failures, don't raise
        profile_enabled=True,
    )
    model = create_llm_model(
        MODEL_ID, verify=True, config=cfg, dtype=DTYPE, device="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Find VerifyLinear instances and corrupt their GPU weights.
    # VerifyLinear is a plain class (not nn.Module), stored as attributes on
    # LlamaMLPVerify / LlamaAttentionVerify modules. Walk all modules and
    # inspect their __dict__ for VerifyLinear instances.
    from verified_core.verify_linear import VerifyLinear

    corrupted_tags = []
    for _name, module in model.named_modules():
        for attr_name in list(vars(module)):
            obj = getattr(module, attr_name, None)
            if isinstance(obj, VerifyLinear):
                with torch.no_grad():
                    obj.linear.weight.add_(
                        torch.randn_like(obj.linear.weight) * 0.5
                    )
                corrupted_tags.append(obj.tag)
                if len(corrupted_tags) >= 3:
                    break
        if len(corrupted_tags) >= 3:
            break

    assert len(corrupted_tags) > 0, "No VerifyLinear layers found"
    print(f"\n[e2e-llm] Corrupted layers: {corrupted_tags}")

    tokenizer, inputs = _tokenize(model, PROMPT)
    _ = model(**inputs)

    runtime = model._verify_runtime
    runtime.flush()

    records = runtime.profiler.records if runtime.profiler else []
    failed = [r for r in records if not r.ok]
    failed_tags = {r.tag for r in failed}

    print(f"[e2e-llm] Total checks: {len(records)}")
    print(f"[e2e-llm] Failed checks: {len(failed)}")
    print(f"[e2e-llm] Failed tags: {failed_tags}")

    runtime.shutdown()
    del model
    torch.cuda.empty_cache()

    # Every corrupted layer must be detected
    for tag in corrupted_tags:
        assert tag in failed_tags, (
            f"Corruption in '{tag}' was NOT detected! Failed tags: {failed_tags}"
        )
    print(f"[e2e-llm] All {len(corrupted_tags)} corrupted layers detected.")


@torch.no_grad()
def test_e2e_llm_profile_report():
    """
    Run a forward pass and verify that profiling produces meaningful records
    covering attention, MLP, and elementwise operations.
    """
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=2,
        mse_threshold=5e-3,
        elementwise_mse_threshold=1e-3,
        fail_on_error=True,
        profile_enabled=True,
    )
    model = create_llm_model(
        MODEL_ID, verify=True, config=cfg, dtype=DTYPE, device="cuda",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer, inputs = _tokenize(model, PROMPT)
    _ = model(**inputs)

    runtime = model._verify_runtime
    runtime.flush()

    records = runtime.profiler.records if runtime.profiler else []
    tags = {r.tag for r in records}

    print(f"\n[e2e-llm] Profile records: {len(records)}")
    print(f"[e2e-llm] Unique tags: {len(tags)}")

    # Expect q/k/v/o projections, qk/kv matmuls, CPU chain non-linear, mlp gate/up/down
    has_attn_proj = any("q" in t or "k" in t or "v" in t for t in tags)
    has_matmul = any("qk" in t or "kv" in t for t in tags)
    has_mlp = any("gate" in t or "up" in t or "down" in t for t in tags)
    # Non-linears (softmax, silu, etc.) are now recomputed inside the CPU chain
    # and recorded under "cpu_chain_nonlinear" rather than individual submit_elementwise.
    op_names = {r.op_name for r in records}
    has_chain = "cpu_chain_nonlinear" in op_names

    n_fail = sum(1 for r in records if not r.ok)

    runtime.shutdown()
    del model
    torch.cuda.empty_cache()

    assert has_attn_proj, f"No attention projection records found. Tags: {tags}"
    assert has_matmul, f"No matmul records found. Tags: {tags}"
    assert has_mlp, f"No MLP records found. Tags: {tags}"
    assert has_chain, f"No CPU chain records found. Op names: {op_names}"
    assert n_fail == 0, f"{n_fail} verification failures in honest model"
    print("[e2e-llm] All operation categories verified in profile.")
