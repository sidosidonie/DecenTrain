"""
Shared verification infrastructure used by both verified_diffusers and verified_llm.

Core components:
  - VerifyConfig: dataclass with all verification knobs
  - VerifyRuntime: async GPU→CPU verification engine (D2H, Freivalds, SLALOM)
  - VerifyProfiler: timing collection and export (CSV, JSON, plot)
  - verify_linear: Freivalds/SLALOM algorithms, VerifyLinear, copy_to_cpu
"""
from verified_core.config import VerifyConfig, default_verify_config, default_diffusion_config
from verified_core.profiler import VerifyProfiler
from verified_core.runtime import VerifyRuntime

__all__ = [
    "VerifyConfig",
    "VerifyRuntime",
    "VerifyProfiler",
    "default_verify_config",
    "default_diffusion_config",
]
