#!/usr/bin/env python3
"""
Calibrate the performance model against real hardware.

Runs actual verified inference on the current machine, collects profiling
data, and saves calibration factors to a YAML file. The perf_analyze tool
reads these factors to produce accurate predictions.

Supports both LLM and diffusion models.

Usage:
  # Calibrate Z-Image diffusion on current GPU
  python tools/calibrate.py --model Z-Image --type diffusion --steps 1

  # Calibrate an LLM
  python tools/calibrate.py --model Qwen2.5-0.5B --type llm --seq-len 32 --gen-tokens 10

  # Calibrate with custom config
  python tools/calibrate.py --model Qwen3.5-9B --type llm --freivalds-k 4 --dtype bf16

  # Save to custom path
  python tools/calibrate.py --model Z-Image --type diffusion -o calibration/my_machine.yaml

  # List available models
  python tools/calibrate.py --list
"""
from __future__ import annotations

import argparse
import gc
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except ImportError:
    print("PyTorch required: pip install torch", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_hardware() -> dict:
    """Detect current machine hardware."""
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
    }

    # CPU
    info["cpu"] = {
        "name": platform.processor() or "unknown",
        "cores_physical": os.cpu_count() or 0,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"]["name"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass

    # GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_gb": round(getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    else:
        info["gpu"] = {"name": "none", "count": 0}

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Runners
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2


def _free(obj):
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class BenchResult:
    model_name: str = ""
    model_type: str = ""
    dtype: str = ""
    # Origin (no verification)
    origin_forward_ms: float = 0
    origin_prefill_ms: float = 0
    origin_decode_avg_ms: float = 0
    origin_step_ms: float = 0  # diffusion
    # Verified
    verified_forward_ms: float = 0
    verified_prefill_ms: float = 0
    verified_decode_avg_ms: float = 0
    verified_step_ms: float = 0  # diffusion
    # Profiler breakdown
    verify_linear_avg_ms: float = 0
    verify_linear_count: int = 0
    verify_matmul_avg_ms: float = 0
    verify_matmul_count: int = 0
    verify_elementwise_avg_ms: float = 0
    verify_elementwise_count: int = 0
    transfer_linear_avg_ms: float = 0
    transfer_matmul_avg_ms: float = 0
    verify_skip_count: int = 0
    total_checks: int = 0
    total_failures: int = 0
    # Config
    freivalds_k: int = 0
    seq_len: int = 0
    gen_tokens: int = 0
    num_steps: int = 0
    batch_size: int = 1
    peak_vram_mb: float = 0


def run_llm_benchmark(
    model_path: str, dtype_str: str = "bf16", seq_len: int = 32,
    gen_tokens: int = 10, freivalds_k: int = 4, warmup: int = 1,
    timed_runs: int = 3, **kwargs,
) -> BenchResult:
    from transformers import AutoTokenizer
    from verified_core.config import VerifyConfig
    from verified_llm.llm_model import create_llm_model

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16 if dtype_str == "fp16" else torch.float32
    result = BenchResult(model_name=model_path, model_type="llm", dtype=dtype_str,
                         freivalds_k=freivalds_k, seq_len=seq_len, gen_tokens=gen_tokens)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "Explain the concept of verified computation in three sentences."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=seq_len,
                       truncation=True, padding="max_length").to("cuda")
    actual_seq = inputs["input_ids"].shape[1]
    result.seq_len = actual_seq

    # ── Origin baseline ──
    print(f"  [1/4] Loading origin model...")
    model = create_llm_model(model_path, verify=False, dtype=dtype, device="cuda", trust_remote_code=True)
    model.eval()

    print(f"  [2/4] Benchmarking origin (warmup={warmup}, runs={timed_runs})...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
        torch.cuda.synchronize()

        fwd_times = []
        for _ in range(timed_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inputs)
            torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000)

    result.origin_forward_ms = sum(fwd_times) / len(fwd_times)

    # Prefill + decode
    gen_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = gen_inputs["input_ids"].clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=generated, use_cache=True)
        torch.cuda.synchronize()
        result.origin_prefill_ms = (time.perf_counter() - t0) * 1000

        past = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(dim=-1)
        decode_times = []
        for _ in range(gen_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - t0) * 1000)
            past = out.past_key_values
            next_tok = out.logits[:, -1:].argmax(dim=-1)

    result.origin_decode_avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0
    _free(model)

    # ── Verified ──
    print(f"  [3/4] Loading verified model (k={freivalds_k})...")
    cfg = VerifyConfig(
        enabled=True, freivalds_k=freivalds_k, mse_threshold=5e-2,
        elementwise_mse_threshold=1e-2, max_workers=4,
        max_verify_tensor_numel=4_000_000, fail_on_error=False,
        profile_enabled=True,
    )
    model = create_llm_model(model_path, verify=True, config=cfg, dtype=dtype,
                             device="cuda", trust_remote_code=True)
    model.eval()
    runtime = model._verify_runtime
    torch.cuda.reset_peak_memory_stats()

    print(f"  [4/4] Benchmarking verified...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
            runtime.flush()
        runtime.clear_errors()
        runtime.profiler._records.clear()
        torch.cuda.synchronize()

        fwd_times = []
        for _ in range(timed_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inputs)
            runtime.flush()
            torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000)

    result.verified_forward_ms = sum(fwd_times) / len(fwd_times)

    # Prefill + decode (verified)
    runtime.profiler._records.clear()
    runtime._errors.clear()

    gen_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = gen_inputs["input_ids"].clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=generated, use_cache=True)
        past = out.past_key_values
        runtime.flush()
        torch.cuda.synchronize()
        result.verified_prefill_ms = (time.perf_counter() - t0) * 1000

        next_tok = out.logits[:, -1:].argmax(dim=-1)
        decode_times = []
        for _ in range(gen_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            runtime.flush()
            torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - t0) * 1000)
            next_tok = out.logits[:, -1:].argmax(dim=-1)

    result.verified_decode_avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0
    result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    _extract_profiler_stats(runtime.profiler, result)
    runtime.shutdown()
    _free(model)
    return result


def run_diffusion_benchmark(
    model_path: str = "Tongyi-MAI/Z-Image", dtype_str: str = "bf16",
    num_steps: int = 1, freivalds_k: int = 4, warmup: int = 0,
    timed_runs: int = 1, **kwargs,
) -> BenchResult:
    from diffusers import DiffusionPipeline
    from verified_core.config import VerifyConfig
    from verified_diffusers.zimage.pipeline import patch_zimage_pipeline

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16 if dtype_str == "fp16" else torch.float32
    result = BenchResult(model_name=model_path, model_type="diffusion", dtype=dtype_str,
                         freivalds_k=freivalds_k, num_steps=num_steps)

    prompt = "A serene mountain landscape at sunset"
    gen = torch.Generator(device="cuda").manual_seed(42)
    pipe_kwargs = dict(
        prompt=prompt, num_inference_steps=num_steps, guidance_scale=3.5,
        max_sequence_length=128, height=512, width=512, output_type="latent",
    )

    # ── Origin baseline ──
    print(f"  [1/4] Loading origin pipeline...")
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    print(f"  [2/4] Benchmarking origin ({num_steps} steps, runs={timed_runs})...")
    for _ in range(warmup):
        _ = pipe(**pipe_kwargs, generator=torch.Generator(device="cuda").manual_seed(0))
    torch.cuda.synchronize()

    step_times = []
    for _ in range(timed_runs):
        gen = torch.Generator(device="cuda").manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe(**pipe_kwargs, generator=gen)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)

    result.origin_step_ms = sum(step_times) / len(step_times) / num_steps
    result.origin_forward_ms = sum(step_times) / len(step_times)
    _free(pipe)

    # ── Verified ──
    print(f"  [3/4] Loading verified pipeline (k={freivalds_k})...")
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    cfg = VerifyConfig(
        enabled=True, freivalds_k=freivalds_k, mse_threshold=5e-3,
        elementwise_mse_threshold=1e-2, max_workers=4,
        max_verify_tensor_numel=4_000_000, fail_on_error=False,
        profile_enabled=True,
    )
    vpipe = patch_zimage_pipeline(pipe, cfg)
    torch.cuda.reset_peak_memory_stats()

    print(f"  [4/4] Benchmarking verified...")
    for _ in range(warmup):
        _ = vpipe(generator=torch.Generator(device="cuda").manual_seed(0), **pipe_kwargs)
    vpipe.runtime.clear_errors()
    vpipe.runtime.profiler._records.clear()
    torch.cuda.synchronize()

    step_times = []
    for _ in range(timed_runs):
        gen = torch.Generator(device="cuda").manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = vpipe(**pipe_kwargs, generator=gen)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)

    result.verified_step_ms = sum(step_times) / len(step_times) / num_steps
    result.verified_forward_ms = sum(step_times) / len(step_times)
    result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    _extract_profiler_stats(vpipe.runtime.profiler, result)
    vpipe.shutdown()
    _free(vpipe)
    return result


def _extract_profiler_stats(profiler, result: BenchResult):
    """Extract per-category timing from profiler records."""
    records = profiler.records
    result.total_checks = len(records)
    result.total_failures = sum(1 for r in records if not r.ok)

    by_cat: Dict[str, List[float]] = {}
    for r in records:
        key = f"{r.category}:{r.op_name}"
        by_cat.setdefault(key, []).append(r.duration_ms)

    def _avg(key_prefix: str) -> tuple:
        times = []
        for k, v in by_cat.items():
            if k.startswith(key_prefix):
                times.extend(v)
        if times:
            return sum(times) / len(times), len(times)
        return 0.0, 0

    result.verify_linear_avg_ms, result.verify_linear_count = _avg("verify:linear")
    result.verify_matmul_avg_ms, result.verify_matmul_count = _avg("verify:matmul")
    result.verify_elementwise_avg_ms, result.verify_elementwise_count = _avg("verify:softmax")
    if result.verify_elementwise_count == 0:
        result.verify_elementwise_avg_ms, result.verify_elementwise_count = _avg("verify:silu")

    result.transfer_linear_avg_ms, _ = _avg("transfer:linear")
    result.transfer_matmul_avg_ms, _ = _avg("transfer:matmul")
    result.verify_skip_count = sum(len(v) for k, v in by_cat.items() if "skip" in k)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Output
# ═══════════════════════════════════════════════════════════════════════════════

def build_calibration_yaml(hw: dict, result: BenchResult) -> dict:
    """Build calibration YAML from benchmark results."""
    cal: Dict[str, Any] = {
        "calibration": {
            "version": 1,
            "timestamp": hw["timestamp"],
            "machine": {
                "cpu": hw["cpu"]["name"],
                "cpu_cores": hw["cpu"]["cores_physical"],
                "gpu": hw["gpu"]["name"],
                "gpu_count": hw["gpu"]["count"],
                "gpu_memory_gb": hw["gpu"]["memory_gb"],
                "pytorch": hw["pytorch"],
            },
            "benchmark": {
                "model": result.model_name,
                "model_type": result.model_type,
                "dtype": result.dtype,
                "freivalds_k": result.freivalds_k,
                "peak_vram_mb": round(result.peak_vram_mb, 0),
            },
        },
    }

    if result.model_type == "llm":
        cal["calibration"]["benchmark"].update({
            "seq_len": result.seq_len,
            "gen_tokens": result.gen_tokens,
        })
        cal["calibration"]["measured"] = {
            "origin": {
                "forward_ms": round(result.origin_forward_ms, 2),
                "prefill_ms": round(result.origin_prefill_ms, 2),
                "decode_avg_ms": round(result.origin_decode_avg_ms, 2),
                "decode_tok_per_sec": round(1000 / result.origin_decode_avg_ms, 1) if result.origin_decode_avg_ms > 0 else 0,
            },
            "verified": {
                "forward_ms": round(result.verified_forward_ms, 2),
                "prefill_ms": round(result.verified_prefill_ms, 2),
                "decode_avg_ms": round(result.verified_decode_avg_ms, 2),
                "decode_tok_per_sec": round(1000 / result.verified_decode_avg_ms, 1) if result.verified_decode_avg_ms > 0 else 0,
                "overhead_forward": round(result.verified_forward_ms / result.origin_forward_ms, 2) if result.origin_forward_ms > 0 else 0,
                "overhead_decode": round(result.verified_decode_avg_ms / result.origin_decode_avg_ms, 2) if result.origin_decode_avg_ms > 0 else 0,
            },
        }
    else:
        cal["calibration"]["benchmark"]["num_steps"] = result.num_steps
        cal["calibration"]["measured"] = {
            "origin": {
                "total_ms": round(result.origin_forward_ms, 2),
                "per_step_ms": round(result.origin_step_ms, 2),
            },
            "verified": {
                "total_ms": round(result.verified_forward_ms, 2),
                "per_step_ms": round(result.verified_step_ms, 2),
                "overhead": round(result.verified_forward_ms / result.origin_forward_ms, 2) if result.origin_forward_ms > 0 else 0,
            },
        }

    # Per-operation breakdown
    cal["calibration"]["profiler"] = {
        "total_checks": result.total_checks,
        "total_failures": result.total_failures,
        "pass_rate_pct": round((result.total_checks - result.total_failures) / max(1, result.total_checks) * 100, 1),
        "skipped": result.verify_skip_count,
        "linear_verify": {
            "count": result.verify_linear_count,
            "avg_ms": round(result.verify_linear_avg_ms, 3),
        },
        "matmul_verify": {
            "count": result.verify_matmul_count,
            "avg_ms": round(result.verify_matmul_avg_ms, 3),
        },
        "elementwise_verify": {
            "count": result.verify_elementwise_count,
            "avg_ms": round(result.verify_elementwise_avg_ms, 3),
        },
        "transfer_linear_avg_ms": round(result.transfer_linear_avg_ms, 3),
        "transfer_matmul_avg_ms": round(result.transfer_matmul_avg_ms, 3),
    }

    # Derived calibration factors for perf_analyze
    # These scale the analytical model to match real measurements
    cal["calibration"]["factors"] = {}
    if result.model_type == "llm" and result.origin_decode_avg_ms > 0:
        cal["calibration"]["factors"]["gpu_decode_ms"] = round(result.origin_decode_avg_ms, 2)
        cal["calibration"]["factors"]["verified_decode_ms"] = round(result.verified_decode_avg_ms, 2)
    elif result.model_type == "diffusion" and result.origin_step_ms > 0:
        cal["calibration"]["factors"]["gpu_step_ms"] = round(result.origin_step_ms, 2)
        cal["calibration"]["factors"]["verified_step_ms"] = round(result.verified_step_ms, 2)

    if result.verify_linear_avg_ms > 0:
        cal["calibration"]["factors"]["linear_verify_ms"] = round(result.verify_linear_avg_ms, 3)
    if result.verify_matmul_avg_ms > 0:
        cal["calibration"]["factors"]["matmul_verify_ms"] = round(result.verify_matmul_avg_ms, 3)
    if result.transfer_linear_avg_ms > 0:
        cal["calibration"]["factors"]["transfer_linear_ms"] = round(result.transfer_linear_avg_ms, 3)
    if result.transfer_matmul_avg_ms > 0:
        cal["calibration"]["factors"]["transfer_matmul_ms"] = round(result.transfer_matmul_avg_ms, 3)

    return cal


def print_report(hw: dict, result: BenchResult):
    """Print human-readable calibration report."""
    W = 90
    print(f"\n{'='*W}")
    print(f"  Calibration Report")
    print(f"{'='*W}")
    print(f"  Machine: {hw['cpu']['name']} ({hw['cpu']['cores_physical']}C) + {hw['gpu']['name']} ({hw['gpu']['memory_gb']}GB)")
    print(f"  PyTorch: {hw['pytorch']}")
    print(f"  Model: {result.model_name} ({result.model_type}) dtype={result.dtype}")

    if result.model_type == "llm":
        print(f"  Config: seq_len={result.seq_len} gen_tokens={result.gen_tokens} k={result.freivalds_k}")
        print(f"\n  {'Metric':<35} {'Origin':>12} {'Verified':>12} {'Overhead':>10}")
        print(f"  {'-'*70}")
        print(f"  {'Forward (ms)':<35} {result.origin_forward_ms:>12.1f} {result.verified_forward_ms:>12.1f} {result.verified_forward_ms/max(0.01,result.origin_forward_ms):>9.1f}x")
        print(f"  {'Prefill (ms)':<35} {result.origin_prefill_ms:>12.1f} {result.verified_prefill_ms:>12.1f} {result.verified_prefill_ms/max(0.01,result.origin_prefill_ms):>9.1f}x")
        print(f"  {'Decode avg (ms/tok)':<35} {result.origin_decode_avg_ms:>12.1f} {result.verified_decode_avg_ms:>12.1f} {result.verified_decode_avg_ms/max(0.01,result.origin_decode_avg_ms):>9.1f}x")
        otps = 1000 / result.origin_decode_avg_ms if result.origin_decode_avg_ms > 0 else 0
        vtps = 1000 / result.verified_decode_avg_ms if result.verified_decode_avg_ms > 0 else 0
        print(f"  {'Decode tok/s':<35} {otps:>12.1f} {vtps:>12.1f}")
    else:
        print(f"  Config: steps={result.num_steps} k={result.freivalds_k}")
        print(f"\n  {'Metric':<35} {'Origin':>12} {'Verified':>12} {'Overhead':>10}")
        print(f"  {'-'*70}")
        print(f"  {'Total (ms)':<35} {result.origin_forward_ms:>12.1f} {result.verified_forward_ms:>12.1f} {result.verified_forward_ms/max(0.01,result.origin_forward_ms):>9.1f}x")
        print(f"  {'Per step (ms)':<35} {result.origin_step_ms:>12.1f} {result.verified_step_ms:>12.1f} {result.verified_step_ms/max(0.01,result.origin_step_ms):>9.1f}x")

    print(f"\n  Verification Breakdown:")
    print(f"    Total checks: {result.total_checks} | Failures: {result.total_failures} | Skipped: {result.verify_skip_count}")
    print(f"    Linear verify:  {result.verify_linear_count:>5} checks, avg {result.verify_linear_avg_ms:.3f} ms")
    print(f"    Matmul verify:  {result.verify_matmul_count:>5} checks, avg {result.verify_matmul_avg_ms:.3f} ms")
    print(f"    Elementwise:    {result.verify_elementwise_count:>5} checks, avg {result.verify_elementwise_avg_ms:.3f} ms")
    print(f"    D2H linear:     avg {result.transfer_linear_avg_ms:.3f} ms")
    print(f"    D2H matmul:     avg {result.transfer_matmul_avg_ms:.3f} ms")
    print(f"    Peak VRAM:      {result.peak_vram_mb:.0f} MB")
    print(f"{'='*W}")


# ═══════════════════════════════════════════════════════════════════════════════
# Available models
# ═══════════════════════════════════════════════════════════════════════════════

KNOWN_MODELS = {
    # LLM
    "Qwen2.5-0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", "llm"),
    "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B-Instruct", "llm"),
    "Qwen3.5-9B": ("Qwen/Qwen3.5-9B", "llm"),
    # Diffusion
    "Z-Image": ("Tongyi-MAI/Z-Image", "diffusion"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate performance model against real hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="Model name or HuggingFace path")
    parser.add_argument("--type", choices=["llm", "diffusion"], help="Model type")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--freivalds-k", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32, help="LLM prompt length")
    parser.add_argument("--gen-tokens", type=int, default=10, help="LLM tokens to generate")
    parser.add_argument("--steps", type=int, default=1, help="Diffusion steps")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("-o", "--output", help="Output calibration YAML path")
    parser.add_argument("--list", action="store_true", help="List known models")
    args = parser.parse_args()

    if args.list:
        print("\nKnown models (use --model <name>):")
        for name, (path, mtype) in sorted(KNOWN_MODELS.items()):
            print(f"  {name:<20} {mtype:<12} {path}")
        print("\nOr pass any HuggingFace model path with --type llm|diffusion")
        return

    if not args.model:
        parser.print_help()
        print("\nError: --model required (use --list to see options)")
        sys.exit(1)

    # Resolve model path
    if args.model in KNOWN_MODELS:
        model_path, model_type = KNOWN_MODELS[args.model]
        if args.type:
            model_type = args.type
    else:
        model_path = args.model
        model_type = args.type or "llm"

    print(f"\n  Calibrating: {model_path} ({model_type})")
    print(f"  dtype={args.dtype} k={args.freivalds_k} warmup={args.warmup} runs={args.runs}")

    hw = detect_hardware()
    print(f"  Machine: {hw['cpu']['name']} + {hw['gpu']['name']}")

    if model_type == "llm":
        result = run_llm_benchmark(
            model_path, dtype_str=args.dtype, seq_len=args.seq_len,
            gen_tokens=args.gen_tokens, freivalds_k=args.freivalds_k,
            warmup=args.warmup, timed_runs=args.runs,
        )
    else:
        result = run_diffusion_benchmark(
            model_path, dtype_str=args.dtype, num_steps=args.steps,
            freivalds_k=args.freivalds_k, warmup=args.warmup, timed_runs=args.runs,
        )

    print_report(hw, result)

    # Save calibration
    cal = build_calibration_yaml(hw, result)
    out_path = args.output
    if not out_path:
        gpu_short = hw["gpu"]["name"].replace(" ", "_").replace("/", "_")
        model_short = args.model.replace("/", "_")
        out_path = f"calibration/{gpu_short}_{model_short}.yaml"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cal, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Calibration saved to: {out_path}")
    print(f"  Use with perf_analyze: add 'calibration: {out_path}' to your system config")


if __name__ == "__main__":
    main()
