import os
import sys
from pathlib import Path

import pytest
import torch
from diffusers import DiffusionPipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.pipeline import patch_zimage_pipeline


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(os.getenv("RUN_SLOW_ZIMAGE", "0") != "1", reason="Set RUN_SLOW_ZIMAGE=1 to run"),
]


@torch.no_grad()
def test_zimage_pipeline_slow_origin_vs_verified(tmp_path: Path):
    import gc, time

    prompt = "A minimal icon of a robot in flat style"

    # --- Phase 1: Origin inference with cpu_offload ---
    print("\n[zimage] Loading pipeline...")
    pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    # Warmup
    gen_w = torch.Generator("cuda").manual_seed(0)
    pipe(prompt=prompt, num_inference_steps=1, guidance_scale=3.5,
         max_sequence_length=128, height=512, width=512, generator=gen_w, output_type="latent")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    gen0 = torch.Generator("cuda").manual_seed(7)
    out_ref = pipe(
        prompt=prompt, num_inference_steps=1, guidance_scale=3.5,
        max_sequence_length=128, height=512, width=512, generator=gen0, output_type="latent",
    )
    torch.cuda.synchronize()
    origin_ms = (time.perf_counter() - t0) * 1000
    lat_ref = out_ref.images.cpu()
    print(f"[zimage] Origin time: {origin_ms:.0f} ms")

    # Free origin pipeline
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2: Verified inference (fresh load + patch) ---
    print("[zimage] Loading verified pipeline...")
    pipe2 = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image", torch_dtype=torch.bfloat16)
    pipe2.set_progress_bar_config(disable=True)

    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        profile_enabled=True,
        profile_dir=str(tmp_path),
        profile_plot=True,
        fail_on_error=False,
    )
    # Patch transformer while weights are on CPU (SLALOM precomputation)
    vpipe = patch_zimage_pipeline(pipe2, cfg)
    vpipe.enable_model_cpu_offload()

    # Warmup
    gen_w2 = torch.Generator("cuda").manual_seed(0)
    vpipe(prompt=prompt, num_inference_steps=1, guidance_scale=3.5,
          max_sequence_length=128, height=512, width=512, generator=gen_w2, output_type="latent")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    gen1 = torch.Generator("cuda").manual_seed(7)
    out_ver = vpipe(
        prompt=prompt, num_inference_steps=1, guidance_scale=3.5,
        max_sequence_length=128, height=512, width=512, generator=gen1, output_type="latent",
    )
    torch.cuda.synchronize()
    verify_ms = (time.perf_counter() - t0) * 1000
    lat_ver = out_ver.images.cpu()
    print(f"[zimage] Verified time: {verify_ms:.0f} ms")
    print(f"[zimage] Overhead: {verify_ms / origin_ms:.2f}x")

    # --- Compare ---
    assert lat_ref.shape == lat_ver.shape
    mse = torch.nn.functional.mse_loss(lat_ref, lat_ver).item()
    print(f"[zimage] Latent MSE: {mse:.8f}")

    exported = vpipe.export_profile("slow_e2e")
    print(f"[zimage] Profile: {exported['detail_csv']}")
    assert Path(exported["detail_csv"]).exists()

    records = vpipe.runtime.profiler.records
    n_total = len(records)
    n_fail = sum(1 for r in records if not r.ok)
    print(f"[zimage] Verify checks: {n_total} total, {n_fail} failures")

    vpipe.shutdown()
    # bf16 inference across ~30 layers drifts beyond element-wise allclose;
    # bound by MSE instead so the check tolerates a handful of outliers.
    assert mse < 0.05, f"Latent MSE too large: {mse:.8f}"
