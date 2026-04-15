"""
Freivalds' algorithm for probabilistic matrix multiplication verification,
and VerifyLinear — a verified linear layer backed by VerifyRuntime.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.nn import Module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random vector cache (reuse across calls for same shape)
# ---------------------------------------------------------------------------

class _RandomVecCache:
    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, n: int, k: int, dtype: torch.dtype) -> torch.Tensor:
        key = f"{n}-{k}-{dtype}"
        if key not in self._cache:
            self._cache[key] = torch.randn((n, k), dtype=dtype, device="cpu")
            logger.debug("Created random vector %s", key)
        return self._cache[key]


_rand_vec_cache = _RandomVecCache()


# ---------------------------------------------------------------------------
# SLALOM preprocessed Freivalds verification (Tramer & Boneh, ICLR 2019)
#
# Key insight: for y = x @ W^T with fixed W, precompute s_tilde = W^T @ s
# offline. Online verification becomes y @ s == x @ s_tilde (two dot
# products) at O(n*k) cost instead of O(n^2*k).
# ---------------------------------------------------------------------------

def slalom_precompute(
    weight_t: torch.Tensor,
    k: int = 10,
    s_range: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute SLALOM verification vectors for a linear layer.

    Follows Tramer & Boneh (ICLR 2019) Lemma 3.1: for y = x @ W^T with
    fixed W, precompute s_tilde = W^T @ s offline. Online verification
    becomes y @ s == x @ s_tilde at O(n*k) cost.

    The original paper uses large integer random vectors (|S| = 2^20) with
    quantized integer arithmetic in a finite field. For floating-point
    inference, we use unit-scale Gaussian random vectors to avoid amplifying
    fp32 rounding errors (floating-point matmul is not associative).

    Args:
        weight_t: Transposed weight matrix W^T, shape [in_features, out_features].
        k: Number of random vectors (repetitions).
        s_range: Unused in Gaussian mode; retained for API compatibility.

    Returns:
        (s, s_tilde) where:
          s: Random vector, shape [out_features, k]
          s_tilde: Preprocessed W^T @ s, shape [in_features, k]
    """
    out_features = weight_t.shape[1]
    # Gaussian random vectors (unit scale) for floating-point compatibility.
    # The paper's integer vectors require exact field arithmetic; Gaussian
    # vectors with MSE threshold provide equivalent probabilistic guarantees
    # in the floating-point setting.
    s = torch.randn(out_features, k, dtype=torch.float32)
    # Precompute s_tilde = W^T @ s in float64 for maximum precision.
    # This is an offline cost; double precision eliminates precomputation
    # rounding error so the online check only reflects GPU computation error.
    s_tilde = torch.matmul(weight_t.double(), s.double()).float()
    return s, s_tilde


def slalom_verify_preprocessed(
    x: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    s_tilde: torch.Tensor,
) -> float:
    """Verify y = x @ W^T using SLALOM preprocessed Freivalds check.

    Online cost: O(batch * seq * (in + out) * k) -- two matrix-vector products.
    Compare with standard Freivalds: O(in * out * k + batch * seq * (in + out) * k).

    Args:
        x: Input tensor, shape [..., in_features], float32
        y: Output tensor, shape [..., out_features], float32
        s: Random vector, shape [out_features, k]
        s_tilde: Preprocessed vector W^T @ s, shape [in_features, k]

    Returns:
        MSE between y @ s and x @ s_tilde.
    """
    y_s = torch.matmul(y, s)          # [..., k]
    x_st = torch.matmul(x, s_tilde)  # [..., k]
    return F.mse_loss(y_s, x_st).item()


# ---------------------------------------------------------------------------
# Noise injection (for threat-model testing)
# ---------------------------------------------------------------------------

def add_noise(C: torch.Tensor, noise_scale: float | None = None) -> torch.Tensor:
    if noise_scale is None or noise_scale == 0:
        return C
    noise = torch.rand(C.shape, device=C.device, dtype=C.dtype) * noise_scale
    return C + noise


# ---------------------------------------------------------------------------
# Freivalds' algorithm variants
# ---------------------------------------------------------------------------

def freivalds_batch_matmul_bias(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    bias: torch.Tensor | None, k: int = 10,
) -> float:
    """Verify C = A @ B + bias (batched, any dim >= 2)."""
    assert A.device == B.device == C.device
    r = _rand_vec_cache.get(C.shape[-1], k, A.dtype)
    Br = torch.matmul(B, r)
    ABr = torch.matmul(A, Br)
    if bias is not None:
        bias_r = torch.matmul(bias.unsqueeze(-2), r)
        ABr = ABr + bias_r.expand_as(ABr)
    Cr = torch.matmul(C, r)
    return F.mse_loss(ABr, Cr).item()


def freivalds_algorithm_2d_bias(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    bias: torch.Tensor | None = None, k: int = 10,
) -> float:
    """Verify C = A @ B + bias (2D matrices)."""
    assert A.device == B.device == C.device
    r = _rand_vec_cache.get(C.shape[-1], k, A.dtype)
    Br = torch.mm(B, r)
    ABr = torch.mm(A, Br)
    if bias is not None:
        bias_r = torch.mm(bias.view(1, -1), r)
        ABr = ABr + bias_r.expand(ABr.shape)
    Cr = torch.mm(C, r)
    return F.mse_loss(ABr, Cr).item()


def freivalds_batch_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, k: int = 10) -> float:
    """Verify C = A @ B (batched, no bias)."""
    assert A.device == B.device == C.device
    r = _rand_vec_cache.get(C.shape[-1], k, A.dtype)
    Br = torch.matmul(B, r)
    ABr = torch.matmul(A, Br)
    Cr = torch.matmul(C, r)
    return F.mse_loss(ABr, Cr).item()


def freivalds_batch_matmul_parallel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, k: int = 10) -> float:
    """Verify C = A @ B with A@Br and C@r computed in parallel."""
    assert A.device == B.device == C.device
    r = _rand_vec_cache.get(C.shape[-1], k, A.dtype)
    Br = torch.matmul(B, r)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_ABr = pool.submit(torch.matmul, A, Br)
        f_Cr = pool.submit(torch.matmul, C, r)
        ABr = f_ABr.result()
        Cr = f_Cr.result()

    return F.mse_loss(ABr, Cr).item()


def freivalds_algorithm_2d(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, k: int = 10) -> float:
    """Verify C = A @ B (2D matrices, no bias)."""
    assert A.device == B.device == C.device
    r = _rand_vec_cache.get(C.shape[-1], k, A.dtype)
    Br = torch.mm(B, r)
    ABr = torch.mm(A, Br)
    Cr = torch.mm(C, r)
    return F.mse_loss(ABr, Cr).item()


def freivalds_algorithm_bias(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, bias: torch.Tensor | None, k: int = 10) -> float:
    """Dispatch to 2D or batched bias variant based on input rank."""
    if len(A.shape) > 2:
        return freivalds_batch_matmul_bias(A, B, C, bias, k)
    elif len(A.shape) == 2:
        return freivalds_algorithm_2d_bias(A, B, C, bias, k)
    else:
        raise ValueError(f"Invalid shape: {A.shape}")


def freivalds_algorithm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, k: int = 10) -> float:
    """Dispatch to 2D or batched variant based on input rank."""
    if len(A.shape) > 2:
        return freivalds_batch_matmul_parallel(A, B, C, k)
    elif len(A.shape) == 2:
        return freivalds_algorithm_2d(A, B, C, k)
    else:
        raise ValueError(f"Invalid shape: {A.shape}")


# ---------------------------------------------------------------------------
# GPU → CPU async copy
# ---------------------------------------------------------------------------

def copy_to_cpu(x_device: torch.Tensor, stream_copy: torch.cuda.Stream):
    """Async copy a GPU tensor to pinned CPU memory on the given stream."""
    if x_device.is_cuda:
        x_host = torch.empty_like(x_device, device="cpu", pin_memory=True)
        e = torch.cuda.Event()
        with torch.cuda.stream(stream_copy):
            x_host.copy_(x_device, non_blocking=True)
            e.record(stream_copy)
        return x_host, e
    return x_device, None


# ---------------------------------------------------------------------------
# VerifyLinear — async-verified linear layer via VerifyRuntime
# ---------------------------------------------------------------------------

class VerifyLinear:
    """Verified linear layer backed by VerifyRuntime.

    Uses SLALOM-style preprocessed Freivalds (Tramer & Boneh, ICLR 2019):
    since weights W are fixed at inference, precompute s_tilde = W^T @ s
    offline. Online verification becomes y @ s == x @ s_tilde at O(n*k)
    cost instead of O(n^2*k).

    Two forward modes:
      - forward(): GPU matmul + auto-submit verification (legacy/diffusers)
      - forward_gpu_only(): GPU matmul only; caller handles verification via
        CPU chain (used by LlamaAttentionVerify / LlamaMLPVerify)
    """

    def __init__(self, linear: torch.nn.Linear, runtime, tag: str = "linear", noise: float | None = None):
        from verified_diffusers.zimage.runtime import VerifyRuntime

        assert isinstance(runtime, VerifyRuntime), "VerifyLinear requires a VerifyRuntime instance"

        self.linear = linear
        self.runtime = runtime
        self.tag = tag
        self.noise = noise
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # SLALOM preprocessing: precompute verification vectors offline.
        cfg = runtime.config
        self.s, self.s_tilde = slalom_precompute(
            linear.weight.detach().t().float().to("cpu"),
            k=cfg.freivalds_k,
            s_range=cfg.slalom_s_range,
        )

        # Bias on CPU for chain verification.
        self.bias_cpu = linear.bias.detach().float().to("cpu") if linear.bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """GPU forward + auto-submit SLALOM verification (legacy mode)."""
        out = F.linear(input, self.linear.weight, bias=None)
        out = add_noise(out, self.noise)
        self.runtime.submit_linear_preprocessed(
            tag=self.tag, x_gpu=input, y_gpu=out,
            s=self.s, s_tilde=self.s_tilde,
        )
        return out

    def forward_gpu_only(self, input: torch.Tensor) -> torch.Tensor:
        """GPU forward only, no verification. Caller handles via CPU chain."""
        out = F.linear(input, self.linear.weight, bias=None)
        out = add_noise(out, self.noise)
        return out

    def add_bias(self, input: torch.Tensor) -> torch.Tensor:
        """Add bias on GPU."""
        if self.linear.bias is not None:
            return input + self.linear.bias
        return input

    def add_bias_cpu(self, input_cpu: torch.Tensor) -> torch.Tensor:
        """Add bias on CPU for chain verification."""
        if self.bias_cpu is not None:
            return input_cpu + self.bias_cpu
        return input_cpu
