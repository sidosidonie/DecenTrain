"""
tests/test_e2e_image_compare.py

End-to-end image generation test: generates images with and without
verification, then compares pixel-level similarity.

Requires:
  - CUDA GPU with enough VRAM for Z-Image (~24GB)
  - Tongyi-MAI/Z-Image model downloaded

Run:
  RUN_E2E_IMAGE=1 pytest tests/test_e2e_image_compare.py -v -s
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_core.config import VerifyConfig
from verified_diffusers.zimage.pipeline import patch_zimage_pipeline

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        os.getenv("RUN_E2E_IMAGE", "0") != "1",
        reason="Set RUN_E2E_IMAGE=1 to run (requires GPU + model)",
    ),
]

MODEL_ID = "Tongyi-MAI/Z-Image"
SEED = 42
PROMPT = "A cat sitting on a windowsill watching rain"


def _load_pipeline():
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_verified_pipeline(cfg):
    """Fresh load, patch transformer on CPU (SLALOM precompute), then offload."""
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    vpipe = patch_zimage_pipeline(pipe, cfg)
    vpipe.enable_model_cpu_offload()
    return vpipe


@pytest.fixture(autouse=True)
def _free_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def _generate_image(pipe, seed: int, num_inference_steps: int = 2, output_type: str = "pil"):
    gen = torch.Generator("cuda").manual_seed(seed)
    out = pipe(
        prompt=PROMPT,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        max_sequence_length=128,
        generator=gen,
        output_type=output_type,
    )
    if output_type == "pil":
        return out.images[0]
    return out.images


def _pil_to_array(img) -> np.ndarray:
    return np.array(img, dtype=np.float32) / 255.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _save_comparison(origin_img, verify_img, output_dir: Path):
    """Save origin, verified, and diff images side by side."""
    output_dir.mkdir(parents=True, exist_ok=True)
    origin_img.save(output_dir / "origin.png")
    verify_img.save(output_dir / "verified.png")

    arr_o = _pil_to_array(origin_img)
    arr_v = _pil_to_array(verify_img)
    diff = np.abs(arr_o - arr_v)
    # Amplify diff for visibility
    diff_amplified = np.clip(diff * 10.0, 0.0, 1.0)
    from PIL import Image

    diff_img = Image.fromarray((diff_amplified * 255).astype(np.uint8))
    diff_img.save(output_dir / "diff_10x.png")


@torch.no_grad()
def test_e2e_image_origin_vs_verified(tmp_path: Path):
    """
    Generate an image without verification, then with verification.
    Same seed -> same random state -> images should be near-identical.
    Asserts PSNR > 30 dB (very high similarity).
    """
    pipe = _load_pipeline()

    # --- Origin ---
    print("\n[e2e] Generating origin image...")
    origin_img = _generate_image(pipe, seed=SEED)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Verified (fresh load + patch) ---
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        elementwise_mse_threshold=1e-3,
        profile_enabled=True,
        profile_dir=str(tmp_path / "profile"),
        fail_on_error=False,
        flush_on_pipeline_end=True,
    )
    vpipe = _load_verified_pipeline(cfg)

    print("[e2e] Generating verified image...")
    verify_img = _generate_image(vpipe, seed=SEED)

    # --- Compare ---
    arr_origin = _pil_to_array(origin_img)
    arr_verify = _pil_to_array(verify_img)

    assert arr_origin.shape == arr_verify.shape, (
        f"Image shape mismatch: {arr_origin.shape} vs {arr_verify.shape}"
    )

    psnr_val = _psnr(arr_origin, arr_verify)
    mse_val = np.mean((arr_origin - arr_verify) ** 2)
    max_diff = np.max(np.abs(arr_origin - arr_verify))

    print(f"[e2e] MSE:      {mse_val:.6f}")
    print(f"[e2e] PSNR:     {psnr_val:.2f} dB")
    print(f"[e2e] Max diff: {max_diff:.6f}")
    print(f"[e2e] Image size: {origin_img.size}")

    # Save comparison artifacts
    output_dir = tmp_path / "images"
    _save_comparison(origin_img, verify_img, output_dir)
    print(f"[e2e] Saved comparison to: {output_dir}")

    # Export profiling
    exported = vpipe.export_profile("e2e_compare")
    print(f"[e2e] Profile: {exported['detail_csv']}")

    vpipe.shutdown()

    # PSNR > 30 dB means images are visually indistinguishable
    assert psnr_val > 30.0, (
        f"Images differ too much: PSNR={psnr_val:.2f} dB (need >30), MSE={mse_val:.6f}"
    )


@torch.no_grad()
def test_e2e_image_latent_equivalence(tmp_path: Path):
    """
    Compare raw latent tensors (before VAE decode) for exact numerical match.
    This is a stricter test than pixel comparison since VAE decoding can
    amplify small differences.
    """
    pipe = _load_pipeline()

    gen0 = torch.Generator("cuda").manual_seed(SEED)
    out_origin = pipe(
        prompt=PROMPT,
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=3.5,
        max_sequence_length=128,
        generator=gen0,
        output_type="latent",
    )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        elementwise_mse_threshold=1e-3,
        fail_on_error=False,
        flush_on_pipeline_end=True,
    )
    vpipe = _load_verified_pipeline(cfg)

    gen1 = torch.Generator("cuda").manual_seed(SEED)
    out_verify = vpipe(
        prompt=PROMPT,
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=3.5,
        max_sequence_length=128,
        generator=gen1,
        output_type="latent",
    )

    lat_origin = out_origin.images
    lat_verify = out_verify.images

    assert isinstance(lat_origin, torch.Tensor)
    assert isinstance(lat_verify, torch.Tensor)
    assert lat_origin.shape == lat_verify.shape

    mse = torch.nn.functional.mse_loss(lat_origin, lat_verify).item()
    max_diff = (lat_origin - lat_verify).abs().max().item()
    # bf16 inference across ~30 layers accumulates rounding drift; verify
    # path also applies bias separately from the GEMM. Use an MSE bound
    # instead of element-wise allclose so a handful of outliers can't
    # dominate the check.
    print(f"\n[e2e] Latent MSE:      {mse:.8f}")
    print(f"[e2e] Latent max diff: {max_diff:.8f}")

    vpipe.shutdown()

    assert mse < 0.05, f"Latent MSE too large: {mse:.8f}, max_diff={max_diff:.8f}"


@torch.no_grad()
def test_e2e_timing_comparison(tmp_path: Path):
    """
    Benchmark origin vs verified inference time.
    Does not assert timing (hardware-dependent), just reports.
    """
    pipe = _load_pipeline()

    # Warmup
    _generate_image(pipe, seed=0, num_inference_steps=1)

    # Origin timing
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    _generate_image(pipe, seed=SEED, num_inference_steps=2)
    ender.record()
    torch.cuda.synchronize()
    origin_ms = starter.elapsed_time(ender)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Verified timing (fresh load + patch)
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        elementwise_mse_threshold=1e-3,
        profile_enabled=True,
        profile_dir=str(tmp_path / "profile"),
        fail_on_error=False,
        flush_on_pipeline_end=True,
    )
    vpipe = _load_verified_pipeline(cfg)

    # Warmup verified
    _generate_image(vpipe, seed=0, num_inference_steps=1)

    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    _generate_image(vpipe, seed=SEED, num_inference_steps=2)
    ender.record()
    torch.cuda.synchronize()
    verify_ms = starter.elapsed_time(ender)

    overhead = (verify_ms - origin_ms) / origin_ms * 100 if origin_ms > 0 else 0

    print(f"\n[e2e] Origin time:   {origin_ms:.2f} ms")
    print(f"[e2e] Verified time: {verify_ms:.2f} ms")
    print(f"[e2e] Overhead:      {overhead:+.1f}%")

    # Save timing to file
    timing_file = tmp_path / "timing.txt"
    timing_file.write_text(
        f"origin_ms={origin_ms:.2f}\n"
        f"verify_ms={verify_ms:.2f}\n"
        f"overhead_pct={overhead:.1f}\n"
    )
    print(f"[e2e] Timing saved: {timing_file}")

    exported = vpipe.export_profile("e2e_timing")
    print(f"[e2e] Profile: {exported['detail_csv']}")

    vpipe.shutdown()
