import copy
import sys
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_core.config import VerifyConfig
from verified_core.runtime import VerifyRuntime
from verified_diffusers.zimage.transformer import VerifiedZImageTransformer2DModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def _drain_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    yield


@torch.no_grad()
def test_verified_transformer_matches_origin_small_model():
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=64,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=32,
        axes_dims=[4, 6, 6],
        axes_lens=[128, 128, 128],
    ).to("cuda")
    model.eval()

    verified_base = copy.deepcopy(model).to("cuda")
    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=1e-4,
        max_workers=2,
        profile_enabled=True,
        fail_on_error=True,
        flush_on_pipeline_end=True,
    )
    runtime = VerifyRuntime(cfg)
    verified_model = VerifiedZImageTransformer2DModel(verified_base, runtime=runtime).to("cuda")
    verified_model.eval()

    x = [
        torch.randn(4, 1, 8, 8, device="cuda", dtype=torch.float32),
        torch.randn(4, 1, 8, 8, device="cuda", dtype=torch.float32),
    ]
    t = torch.tensor([10.0, 30.0], device="cuda", dtype=torch.float32)
    cap_feats = [
        torch.randn(5, 32, device="cuda", dtype=torch.float32),
        torch.randn(7, 32, device="cuda", dtype=torch.float32),
    ]

    out_ref = model(x=x, t=t, cap_feats=cap_feats, return_dict=True).sample
    out_ver = verified_model(x=x, t=t, cap_feats=cap_feats, return_dict=True).sample
    runtime.flush()

    assert len(out_ref) == len(out_ver)
    for a, b in zip(out_ref, out_ver):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)

    assert runtime.pending_tasks == 0
    runtime.shutdown()
