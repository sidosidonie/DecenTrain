"""Backward-compat re-export — canonical location is verified_core.verify_linear."""
from verified_core.verify_linear import *  # noqa: F401,F403
from verified_core.verify_linear import (
    VerifyLinear,
    add_noise,
    copy_to_cpu,
    freivalds_algorithm,
    freivalds_algorithm_2d,
    freivalds_algorithm_2d_bias,
    freivalds_algorithm_bias,
    freivalds_batch_matmul,
    freivalds_batch_matmul_bias,
    freivalds_batch_matmul_parallel,
    slalom_precompute,
    slalom_verify_preprocessed,
)
