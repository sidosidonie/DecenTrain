from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from verified_llm.verify_linear import (
    copy_to_cpu,
    freivalds_algorithm,
    freivalds_algorithm_bias,
    slalom_verify_preprocessed,
)
from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.profiler import VerifyProfiler


@dataclass
class VerifyTaskResult:
    tag: str
    loss: float
    ok: bool
    message: str = ""


class VerifyRuntime:
    def __init__(self, config: VerifyConfig, profiler: Optional[VerifyProfiler] = None):
        self.config = config
        self.profiler = profiler or VerifyProfiler(enabled=config.profile_enabled)
        self.compute_stream = torch.cuda.default_stream()
        self.copy_stream = torch.cuda.Stream()
        self._executor = ThreadPoolExecutor(max_workers=max(1, config.max_workers))
        self._futures: List[Future] = []
        self._op_index = 0
        self._errors: List[str] = []

    def shutdown(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)

    def _enqueue(self, fn) -> None:
        self._futures.append(self._executor.submit(fn))

    def _record_cuda_span(self, category: str, name: str, tag: str, start_event, end_event, shape: str) -> None:
        if not self.config.profile_enabled:
            return

        def _collector():
            end_event.synchronize()
            duration = start_event.elapsed_time(end_event)
            self.profiler.add(category, name, duration, tag=tag, shape=shape)

        self._enqueue(_collector)

    def _next_op_id(self) -> int:
        self._op_index += 1
        return self._op_index

    def d2h_async(self, tensor_gpu: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.cuda.Event]]:
        """Async copy a single GPU tensor to pinned CPU memory.

        Returns (cpu_tensor, done_event). The caller must keep tensor_gpu
        alive until done_event is synchronized (to prevent CUDA caching
        allocator from reclaiming the GPU buffer before D2H completes).
        """
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            cpu_tensor, evt = copy_to_cpu(tensor_gpu, self.copy_stream)
            done = torch.cuda.Event()
            done.record(self.copy_stream)
        return cpu_tensor, done

    def should_verify_now(self) -> bool:
        return self.config.should_verify(self._next_op_id())

    def submit_linear(
        self,
        tag: str,
        x_gpu: torch.Tensor,
        y_gpu: torch.Tensor,
        weight_t_cpu: torch.Tensor,
        bias_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.config.enabled or not self.should_verify_now():
            return
        max_numel = self.config.max_verify_tensor_numel
        if x_gpu.numel() > max_numel or y_gpu.numel() > max_numel:
            self.profiler.add(
                "verify_skip",
                "linear_too_large",
                0.0,
                tag=tag,
                shape=str(tuple(y_gpu.shape)),
                ok=True,
                extra=f"max_numel={max_numel}",
            )
            return

        copy_st = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        copy_ed = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            if self.config.profile_enabled:
                copy_st.record(self.copy_stream)
            x_cpu, ex = copy_to_cpu(x_gpu, self.copy_stream)
            y_cpu, ey = copy_to_cpu(y_gpu, self.copy_stream)
            done_event = torch.cuda.Event()
            done_event.record(self.copy_stream)
            if self.config.profile_enabled:
                copy_ed.record(self.copy_stream)

        if self.config.profile_enabled:
            self._record_cuda_span("transfer", "linear_d2h", tag, copy_st, copy_ed, str(tuple(y_gpu.shape)))

        k = self.config.freivalds_k
        threshold = self.config.mse_threshold
        # Keep GPU tensor references alive in the closure so the CUDA caching
        # allocator cannot reclaim their memory before the async D2H copy on
        # copy_stream completes.  The references are released when _verify()
        # finishes and the closure is garbage-collected.
        _x_gpu_ref = x_gpu
        _y_gpu_ref = y_gpu

        def _verify():
            t0 = time.perf_counter()
            done_event.synchronize()
            if ex is not None:
                ex.synchronize()
            if ey is not None:
                ey.synchronize()
            # Release GPU references now that D2H copy is confirmed complete.
            nonlocal _x_gpu_ref, _y_gpu_ref
            del _x_gpu_ref, _y_gpu_ref
            x_chk = x_cpu.float()
            w_chk = weight_t_cpu.float()
            y_chk = y_cpu.float()
            b_chk = bias_cpu.float() if bias_cpu is not None else None
            if bias_cpu is None:
                loss = freivalds_algorithm(x_chk, w_chk, y_chk, k=k)
            else:
                loss = freivalds_algorithm_bias(x_chk, w_chk, y_chk, b_chk, k=k)
            t1 = time.perf_counter()
            ok = loss <= threshold
            extra_msg = f"loss={loss:.6e}"
            if not math.isfinite(loss):
                # Rare numeric corner case: fallback to exact compare with NaN-aware semantics.
                y_ref = torch.matmul(x_chk, w_chk)
                if b_chk is not None:
                    y_ref = y_ref + b_chk
                atol = max(threshold, 1e-3)
                ok = torch.allclose(y_ref, y_chk, atol=atol, rtol=1e-4, equal_nan=True)
                diff = torch.nan_to_num(y_ref - y_chk, nan=0.0, posinf=0.0, neginf=0.0)
                loss = float(torch.mean(diff * diff).item())
                extra_msg = f"loss=nan,fallback_mse={loss:.6e}"
            self.profiler.add(
                "verify",
                "linear_freivalds",
                (t1 - t0) * 1000.0,
                tag=tag,
                shape=str(tuple(y_gpu.shape)),
                ok=ok,
                extra=extra_msg,
            )
            if not ok:
                msg = f"{tag} verify failed: loss={loss:.6e} threshold={threshold:.6e}"
                self._errors.append(msg)
            return VerifyTaskResult(tag=tag, loss=loss, ok=ok)

        self._enqueue(_verify)

    def submit_linear_preprocessed(
        self,
        tag: str,
        x_gpu: torch.Tensor,
        y_gpu: torch.Tensor,
        s: torch.Tensor,
        s_tilde: torch.Tensor,
    ) -> None:
        """SLALOM preprocessed Freivalds verification (Tramer & Boneh, ICLR 2019).

        Instead of O(n^2*k) online Freivalds (computing W @ r each time),
        uses precomputed s_tilde = W^T @ s. Online cost is O(n*k): two
        matrix-vector products y @ s and x @ s_tilde.
        """
        if not self.config.enabled or not self.should_verify_now():
            return
        max_numel = self.config.max_verify_tensor_numel
        if x_gpu.numel() > max_numel or y_gpu.numel() > max_numel:
            self.profiler.add(
                "verify_skip", "linear_too_large", 0.0,
                tag=tag, shape=str(tuple(y_gpu.shape)), ok=True,
                extra=f"max_numel={max_numel}",
            )
            return

        copy_st = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        copy_ed = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            if self.config.profile_enabled:
                copy_st.record(self.copy_stream)
            x_cpu, ex = copy_to_cpu(x_gpu, self.copy_stream)
            y_cpu, ey = copy_to_cpu(y_gpu, self.copy_stream)
            done_event = torch.cuda.Event()
            done_event.record(self.copy_stream)
            if self.config.profile_enabled:
                copy_ed.record(self.copy_stream)

        if self.config.profile_enabled:
            self._record_cuda_span("transfer", "linear_d2h", tag, copy_st, copy_ed, str(tuple(y_gpu.shape)))

        threshold = self.config.mse_threshold
        _x_gpu_ref = x_gpu
        _y_gpu_ref = y_gpu

        def _verify():
            t0 = time.perf_counter()
            done_event.synchronize()
            if ex is not None:
                ex.synchronize()
            if ey is not None:
                ey.synchronize()
            nonlocal _x_gpu_ref, _y_gpu_ref
            del _x_gpu_ref, _y_gpu_ref

            loss = slalom_verify_preprocessed(x_cpu.float(), y_cpu.float(), s, s_tilde)
            t1 = time.perf_counter()
            ok = loss <= threshold
            extra_msg = f"loss={loss:.6e}"
            if not math.isfinite(loss):
                ok = False
                extra_msg = f"loss=nan"
            self.profiler.add(
                "verify", "linear_slalom", (t1 - t0) * 1000.0,
                tag=tag, shape=str(tuple(y_gpu.shape)), ok=ok, extra=extra_msg,
            )
            if not ok:
                msg = f"{tag} SLALOM verify failed: loss={loss:.6e} threshold={threshold:.6e}"
                self._errors.append(msg)
            return VerifyTaskResult(tag=tag, loss=loss, ok=ok)

        self._enqueue(_verify)

    def submit_matmul(self, tag: str, a_gpu: torch.Tensor, b_gpu: torch.Tensor, c_gpu: torch.Tensor) -> None:
        if not self.config.enabled or not self.should_verify_now():
            return
        max_numel = self.config.max_verify_tensor_numel
        if a_gpu.numel() > max_numel or b_gpu.numel() > max_numel or c_gpu.numel() > max_numel:
            self.profiler.add(
                "verify_skip",
                "matmul_too_large",
                0.0,
                tag=tag,
                shape=str(tuple(c_gpu.shape)),
                ok=True,
                extra=f"max_numel={max_numel}",
            )
            return

        copy_st = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        copy_ed = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            if self.config.profile_enabled:
                copy_st.record(self.copy_stream)
            a_cpu, ea = copy_to_cpu(a_gpu, self.copy_stream)
            b_cpu, eb = copy_to_cpu(b_gpu, self.copy_stream)
            c_cpu, ec = copy_to_cpu(c_gpu, self.copy_stream)
            done_event = torch.cuda.Event()
            done_event.record(self.copy_stream)
            if self.config.profile_enabled:
                copy_ed.record(self.copy_stream)

        if self.config.profile_enabled:
            self._record_cuda_span("transfer", "matmul_d2h", tag, copy_st, copy_ed, str(tuple(c_gpu.shape)))

        k = self.config.freivalds_k
        threshold = self.config.mse_threshold
        # Keep GPU tensor references alive until D2H copy completes.
        _a_gpu_ref = a_gpu
        _b_gpu_ref = b_gpu
        _c_gpu_ref = c_gpu

        def _verify():
            t0 = time.perf_counter()
            done_event.synchronize()
            if ea is not None:
                ea.synchronize()
            if eb is not None:
                eb.synchronize()
            if ec is not None:
                ec.synchronize()
            nonlocal _a_gpu_ref, _b_gpu_ref, _c_gpu_ref
            del _a_gpu_ref, _b_gpu_ref, _c_gpu_ref
            a_chk = a_cpu.float()
            b_chk = b_cpu.float()
            c_chk = c_cpu.float()
            loss = freivalds_algorithm(a_chk, b_chk, c_chk, k=k)
            t1 = time.perf_counter()
            ok = loss <= threshold
            extra_msg = f"loss={loss:.6e}"
            if not math.isfinite(loss):
                c_ref = torch.matmul(a_chk, b_chk)
                atol = max(threshold, 1e-3)
                ok = torch.allclose(c_ref, c_chk, atol=atol, rtol=1e-4, equal_nan=True)
                diff = torch.nan_to_num(c_ref - c_chk, nan=0.0, posinf=0.0, neginf=0.0)
                loss = float(torch.mean(diff * diff).item())
                extra_msg = f"loss=nan,fallback_mse={loss:.6e}"
            self.profiler.add(
                "verify",
                "matmul_freivalds",
                (t1 - t0) * 1000.0,
                tag=tag,
                shape=str(tuple(c_gpu.shape)),
                ok=ok,
                extra=extra_msg,
            )
            if not ok:
                msg = f"{tag} verify failed: loss={loss:.6e} threshold={threshold:.6e}"
                self._errors.append(msg)
            return VerifyTaskResult(tag=tag, loss=loss, ok=ok)

        self._enqueue(_verify)

    _ELEMENTWISE_OPS: Dict[str, Callable] = {
        "softmax": lambda x: F.softmax(x, dim=-1, dtype=torch.float32),
        "silu": lambda x: F.silu(x),
    }

    def submit_elementwise(
        self,
        tag: str,
        input_gpu: torch.Tensor,
        output_gpu: torch.Tensor,
        op_name: str,
    ) -> None:
        """Verify element-wise GPU ops by recomputing on CPU."""
        if not self.config.enabled or not self.should_verify_now():
            return
        if op_name not in self._ELEMENTWISE_OPS:
            raise ValueError(f"Unknown elementwise op: {op_name}")

        max_numel = self.config.max_verify_tensor_numel
        if input_gpu.numel() > max_numel or output_gpu.numel() > max_numel:
            self.profiler.add(
                "verify_skip", f"{op_name}_too_large", 0.0,
                tag=tag, shape=str(tuple(output_gpu.shape)), ok=True,
                extra=f"max_numel={max_numel}",
            )
            return

        copy_st = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        copy_ed = torch.cuda.Event(enable_timing=self.config.profile_enabled)
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            if self.config.profile_enabled:
                copy_st.record(self.copy_stream)
            in_cpu, e_in = copy_to_cpu(input_gpu, self.copy_stream)
            out_cpu, e_out = copy_to_cpu(output_gpu, self.copy_stream)
            done_event = torch.cuda.Event()
            done_event.record(self.copy_stream)
            if self.config.profile_enabled:
                copy_ed.record(self.copy_stream)

        if self.config.profile_enabled:
            self._record_cuda_span("transfer", f"{op_name}_d2h", tag, copy_st, copy_ed, str(tuple(output_gpu.shape)))

        threshold = self.config.elementwise_mse_threshold
        op_fn = self._ELEMENTWISE_OPS[op_name]
        # Keep GPU tensor references alive until D2H copy completes.
        _in_gpu_ref = input_gpu
        _out_gpu_ref = output_gpu

        def _verify():
            t0 = time.perf_counter()
            done_event.synchronize()
            if e_in is not None:
                e_in.synchronize()
            if e_out is not None:
                e_out.synchronize()
            nonlocal _in_gpu_ref, _out_gpu_ref
            del _in_gpu_ref, _out_gpu_ref
            recomputed = op_fn(in_cpu.float())
            out_chk = out_cpu.float()
            loss = F.mse_loss(recomputed, out_chk).item()
            t1 = time.perf_counter()
            ok = loss <= threshold
            extra_msg = f"loss={loss:.6e}"
            self.profiler.add(
                "verify", f"{op_name}_recompute", (t1 - t0) * 1000.0,
                tag=tag, shape=str(tuple(output_gpu.shape)), ok=ok, extra=extra_msg,
            )
            if not ok:
                msg = f"{tag} {op_name} verify failed: loss={loss:.6e} threshold={threshold:.6e}"
                self._errors.append(msg)
            return VerifyTaskResult(tag=tag, loss=loss, ok=ok)

        self._enqueue(_verify)

    def profile_cuda_compute(self, tag: str, op_name: str, shape: str, fn):
        if not self.config.profile_enabled:
            return fn()
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record(self.compute_stream)
        out = fn()
        ed.record(self.compute_stream)
        self._record_cuda_span("compute", op_name, tag, st, ed, shape)
        return out

    def flush(self) -> None:
        pending = self._futures
        self._futures = []
        for fut in pending:
            fut.result()
        if self._errors and self.config.fail_on_error:
            message = "\n".join(self._errors)
            self._errors = []
            raise RuntimeError(f"Verification failed:\n{message}")

    def clear_errors(self) -> None:
        self._errors.clear()

    @property
    def pending_tasks(self) -> int:
        return len(self._futures)
