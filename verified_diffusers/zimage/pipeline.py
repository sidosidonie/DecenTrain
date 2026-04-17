from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline

from verified_core.config import VerifyConfig, default_verify_config
from verified_core.profiler import VerifyProfiler
from verified_core.runtime import VerifyRuntime
from verified_diffusers.zimage.transformer import VerifiedZImageTransformer2DModel


class VerifiedZImagePipeline:
    def __init__(self, pipe: DiffusionPipeline, runtime: VerifyRuntime, profiler: VerifyProfiler, config: VerifyConfig):
        self.pipe = pipe
        self.runtime = runtime
        self.profiler = profiler
        self.config = config

    def __getattr__(self, name: str):
        return getattr(self.pipe, name)

    def __call__(self, *args, **kwargs):
        self.runtime.clear_errors()
        out = self.pipe(*args, **kwargs)
        if self.config.flush_on_pipeline_end:
            self.runtime.flush()
        return out

    def export_profile(self, prefix: Optional[str] = None) -> Dict[str, str]:
        base = prefix or self.config.profile_prefix
        detail_csv = self.config.profile_path(f"{base}_detail.csv")
        summary_csv = self.config.profile_path(f"{base}_summary.csv")
        plot_path = self.config.profile_path(f"{base}_plot.png")
        self.profiler.export_csv(detail_csv, summary_csv)
        if self.config.profile_plot:
            self.profiler.plot(plot_path)
        return {"detail_csv": detail_csv, "summary_csv": summary_csv, "plot": plot_path}

    def shutdown(self) -> None:
        self.runtime.shutdown()


def patch_zimage_pipeline(pipe: DiffusionPipeline, config: Optional[VerifyConfig] = None) -> VerifiedZImagePipeline:
    cfg = config or default_verify_config()
    profiler = VerifyProfiler(enabled=cfg.profile_enabled)
    runtime = VerifyRuntime(cfg, profiler=profiler)
    pipe.transformer = VerifiedZImageTransformer2DModel(pipe.transformer, runtime=runtime)
    return VerifiedZImagePipeline(pipe=pipe, runtime=runtime, profiler=profiler, config=cfg)


def create_verified_zimage_pipeline(
    model_id: str = "Tongyi-MAI/Z-Image",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    verify_config: Optional[VerifyConfig] = None,
    **from_pretrained_kwargs,
) -> VerifiedZImagePipeline:
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
    return patch_zimage_pipeline(pipe, verify_config)
