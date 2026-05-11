"""
tests/conftest.py

Environment compatibility shims applied before any test file imports.

Two system-level incompatibilities are fixed here:

1. torchvision._meta_registrations: tries to register a fake for
   'torchvision::nms' which doesn't exist in torch >= 2.9.  Stubbing the
   module prevents the registration from running while still letting the rest
   of torchvision (transforms, etc.) load normally.

2. flash_attn_2_cuda: the flash-attn C extension was compiled against
   GLIBC_2.32 which is not available on this system.  Stubbing only the C
   extension module (not flash_attn itself) lets flash_attn's Python layer
   import cleanly; diffusers then auto-detects no working flash-attn backend
   and falls back to the native SDPA implementation.
"""
import sys
import types

# ---------------------------------------------------------------------------
# Fix 1: stub torchvision._meta_registrations
# ---------------------------------------------------------------------------
if "torchvision._meta_registrations" not in sys.modules:
    _tv_meta_stub = types.ModuleType("torchvision._meta_registrations")
    sys.modules["torchvision._meta_registrations"] = _tv_meta_stub

# ---------------------------------------------------------------------------
# Fix 2: stub the flash_attn C extension that fails with a GLIBC version error.
# flash_attn itself (the Python package) imports this .so at the top of
# flash_attn/flash_attn_interface.py.  Providing a dummy module object lets
# the Python import chain complete; actual CUDA kernel calls are never reached
# in these tests because diffusers' dispatch_attention_fn falls back to
# PyTorch native SDPA when the C extension has no real implementation.
# ---------------------------------------------------------------------------
if "flash_attn_2_cuda" not in sys.modules:
    _fa_cuda_stub = types.ModuleType("flash_attn_2_cuda")
    sys.modules["flash_attn_2_cuda"] = _fa_cuda_stub


# ---------------------------------------------------------------------------
# Register custom pytest markers.
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "perf: performance tests (gated, run explicitly)"
    )
