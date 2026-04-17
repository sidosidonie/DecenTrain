from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerifyConfig:
    enabled: bool = True
    # SLALOM uses k=2 with |S|=2^20 for 40-bit soundness in integer fields.
    # For floating-point with Gaussian random vectors, we need more repetitions
    # since |S| is effectively continuous. k=10 gives error prob < 2^{-10}.
    freivalds_k: int = 10
    mse_threshold: float = 1e-5
    # Separate threshold for attention matmuls (Q@K^T, P@V) which accumulate
    # more rounding error from bf16 chain (norm, RoPE, softmax before matmul).
    matmul_mse_threshold: float = 0.0  # 0 = use mse_threshold
    matmul_freivalds_k: int = 0  # 0 = use freivalds_k; set lower for faster attention checks
    verify_every_n: int = 1
    max_workers: int = 2
    fail_on_error: bool = True

    # SLALOM integer random vector range: s in [-s_range, s_range].
    # Default 2^19 gives |S| = 2^20 ~ 1M, so per-check soundness = 1/|S|^k.
    slalom_s_range: int = 2**19

    profile_enabled: bool = True
    profile_dir: str = "output/zimage_verify_profile"
    profile_plot: bool = True
    profile_prefix: str = "zimage"

    # Runtime switches
    flush_each_layer: bool = False
    flush_on_pipeline_end: bool = True
    debug: bool = False
    # Threshold for element-wise op verification (softmax, silu, etc.)
    elementwise_mse_threshold: float = 1e-4
    # Skip verification for very large tensors to avoid invalid pinned-memory allocations.
    max_verify_tensor_numel: int = 2_000_000

    def should_verify(self, op_index: int) -> bool:
        if not self.enabled:
            return False
        if self.verify_every_n <= 1:
            return True
        return (op_index % self.verify_every_n) == 0

    def profile_path(self, filename: str) -> str:
        return str(Path(self.profile_dir) / filename)


def default_verify_config(**overrides) -> VerifyConfig:
    cfg = VerifyConfig()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def default_diffusion_config(**overrides) -> VerifyConfig:
    """Defaults tuned for bf16 diffusion models (relative MSE, more workers)."""
    cfg = VerifyConfig(
        mse_threshold=1e-4,            # relative MSE: 1e-4 is ~100x margin over bf16 noise
        matmul_mse_threshold=1e-3,     # matmul accumulates more error through norm/RoPE chain
        elementwise_mse_threshold=1e-2,
        max_workers=12,
        freivalds_k=4,
        max_verify_tensor_numel=4_000_000,
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
