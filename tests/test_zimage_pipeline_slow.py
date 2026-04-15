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
    pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image", dtype=torch.bfloat16, device_map="cuda")
    pipe.set_progress_bar_config(disable=True)

    prompt = "A minimal icon of a robot in flat style"
    gen0 = torch.Generator("cuda").manual_seed(7)
    out_ref = pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=3.5,
        max_sequence_length=128,
        generator=gen0,
        output_type="latent",
    )

    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-3,
        profile_enabled=True,
        profile_dir=str(tmp_path),
        profile_plot=True,
        fail_on_error=True,
    )
    vpipe = patch_zimage_pipeline(pipe, cfg)
    gen1 = torch.Generator("cuda").manual_seed(7)
    out_ver = vpipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=3.5,
        max_sequence_length=128,
        generator=gen1,
        output_type="latent",
    )

    lat_ref = out_ref.images
    lat_ver = out_ver.images
    assert isinstance(lat_ref, torch.Tensor)
    assert isinstance(lat_ver, torch.Tensor)
    assert lat_ref.shape == lat_ver.shape
    assert torch.allclose(lat_ref, lat_ver, atol=1e-2, rtol=1e-2)

    exported = vpipe.export_profile("slow_e2e")
    assert Path(exported["detail_csv"]).exists()
    assert Path(exported["summary_csv"]).exists()
    assert Path(exported["plot"]).exists()
    vpipe.shutdown()
