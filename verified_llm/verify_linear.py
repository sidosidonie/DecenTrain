import enum
from torch.nn import Linear, Module, Parameter
from torch.nn import functional as F, init
import math
import pandas as pd
import torch
from torch import Tensor
from verified_llm.utils.log_utils import g_logger
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

all_matmul = {}

threshold = 1e-5

class GlobRandomVec:

    def __init__(self):
        self.vec = {}

    def get_key(self, n, k, dtype):
        return f"{n}-{k}-{dtype}"

    def get_or_create_vec(self, n, k, dtype):
        key = self.get_key(n, k, dtype)
        if key in self.vec.keys():
            return self.vec[key]
        else:
            r_vec = torch.randn((n, k), dtype=dtype, device="cpu")
            g_logger.info("Create rand vec " + key)
            self.vec[key] = r_vec
            return self.vec[key]

glob_random_vec = GlobRandomVec()

def add_noise(C, noise_scale=None):
    if noise_scale is None or noise_scale == 0:
        return C

    noise = torch.rand(C.shape, device=C.device, dtype=C.dtype) * (noise_scale - noise_scale/1000.)
    g_logger.info(f"Add noise {noise_scale=}, {noise.shape=}, {C.shape=}")
    CC = C + noise
    return CC

def freivalds_batch_matmul_bias(A, B, C, bias, k=10):
    """_summary_

    AxB+b = C

    given C
    AxBxr + bxr = Cxr
    
    [mxk] x [kxn] + [n] = [mxn] 

    """
    # g_logger.info(f"freivalds_batch_matmul: {A.shape=}, {B.shape=}, {C.shape=}")
    assert A.device == B.device == C.device, "All tensors must be on the same device"
    n = C.shape[-1]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
    Br = torch.matmul(B, r)
    ABr = torch.matmul(A, Br)
    if bias is not None:
        # bias shape [..., p] broadcast to last dim
        # Compute bias @ r -> [..., k]
        bias_r = torch.matmul(bias.unsqueeze(-2), r)  # [..., 1, k]
        # Broadcast to match ABr shape
        ABr = ABr + bias_r.expand_as(ABr)

    Cr = torch.matmul(C, r)
    loss = F.mse_loss(ABr, Cr).item()
    #assert loss < threshold, f"Freivalds' algorithm failed with loss {loss}"
    return loss

def freivalds_algorithm_2d_bias(A, B, C, bias=None, k=10):
    """
    Probabilistically verify C = A @ B + bias using Freivalds algorithm (2D matrices)
    A: [m, n]
    B: [n, p]
    C: [m, p]
    bias: None or [p] (row bias)
    k: number of iterations
    """
    g_logger.debug(f"freivalds_algorithm_2d_bias: {A.shape=}, {B.shape=}, {C.shape=}, {bias.shape if bias is not None else None=}")

    # Device check
    assert A.device == B.device == C.device, "All tensors must be on the same device"

    n = C.shape[-1]

    # Get random {0,1} vector(s) of shape [p, k]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)

    # Compute B @ r
    Br = torch.mm(B, r)          # [n, k]

    # Compute A @ (B @ r)
    ABr = torch.mm(A, Br)        # [m, k]

    # Add bias if provided
    if bias is not None:
        # bias: [p] -> [1, p] @ [p, k] = [1, k]
        bias_r = torch.mm(bias.view(1, -1), r)  # [1, k]
        # Broadcast to [m, k]
        ABr = ABr + bias_r.expand(ABr.shape)

    # Compute C @ r
    Cr = torch.mm(C, r)          # [m, k]

    # Compute MSE loss
    loss = F.mse_loss(ABr, Cr).item()

    # Optional threshold check
    # if loss > threshold:
    #     g_logger.fatal(f"Freivalds' algorithm failed with loss {loss}")
    #     print(ABr)
    #     print(Cr)

    return loss

def freivalds_batch_matmul(A, B, C, k=10):
    g_logger.debug(f"freivalds_batch_matmul: {A.shape=}, {B.shape=}, {C.shape=}")
    assert A.device == B.device == C.device, "All tensors must be on the same device"
    n = C.shape[-1]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
    Br = torch.matmul(B, r)
    ABr = torch.matmul(A, Br)
    Cr = torch.matmul(C, r)
    loss = F.mse_loss(ABr, Cr).item()
    #assert loss < threshold, f"Freivalds' algorithm failed with loss {loss}"
    return loss

def freivalds_batch_matmul_parallel(A, B, C, k=10):
    g_logger.debug(f"freivalds_batch_matmul_parallel: {A.shape=}, {B.shape=}, {C.shape=}")
    assert A.device == B.device == C.device, "All tensors must be on the same device"

    n = C.shape[-1]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)

    # Step 1: B @ r (must happen first)
    Br = torch.matmul(B, r)

    # Step 2: Parallelize A @ Br and C @ r
    def compute_ABr(): 
        return torch.matmul(A, Br)

    def compute_Cr(): 
        return torch.matmul(C, r)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_ABr = executor.submit(compute_ABr)
        f_Cr = executor.submit(compute_Cr)
        ABr = f_ABr.result()
        Cr = f_Cr.result()

    # Step 3: compute loss
    loss = F.mse_loss(ABr, Cr).item()
    # assert loss < threshold, f"Freivalds' algorithm failed with loss {loss}"
    return loss

def freivalds_algorithm_2d(A, B, C, k=10):
    g_logger.debug(f"freivalds_algorithm_2d: {A.shape=}, {B.shape=}, {C.shape=}")
    assert A.device == B.device == C.device, "All tensors must be on the same device"
    n = C.shape[-1]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
    Br = torch.mm(B, r)
    ABr = torch.mm(A, Br)
    Cr = torch.mm(C, r)
    loss = F.mse_loss(ABr, Cr).item()
    # if loss > threshold:
    #     g_logger.fatal(f"Freivalds' algorithm failed with loss {loss}")
        # print(ABr)
        # print(Cr)
    return loss

def freivalds_algorithm_bias(A, B, C, bias, k = 10):
    if len(A.shape) > 2:
        return freivalds_batch_matmul_bias(A, B, C, bias, k)
    elif len(A.shape) == 2:
        return freivalds_algorithm_2d_bias(A, B, C, bias, k)
    else:
        raise ValueError(f"Invalid shape: {A.shape}")


def freivalds_algorithm(A, B, C, k = 10):
    if len(A.shape) > 2:
        return freivalds_batch_matmul_parallel(A, B, C, k)
    elif len(A.shape) == 2:
        return freivalds_algorithm_2d(A, B, C, k)
    else:
        raise ValueError(f"Invalid shape: {A.shape}")

def freivalds_algorithm_stream(A, B, C, stream, e1=None, e2=None, e3=None, k=10):
    with torch.cuda.stream(stream):
        n = C.shape[-1]
        r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
        # r = torch.randn((n, k), dtype=torch.float16, device=A.device)
        if len(A.shape) > 2:
            # change A from (n, n, k) to (n*n, k)
            # A = A.reshape(-1, A.shape[-1])
            # B = B.reshape(-1, B.shape[-1])
            # C = C.reshape(-1, C.shape[-1])
            if e2 is not None:
                e2.synchronize()
            Br = torch.matmul(B, r)

            if e1 is not None:
                e1.synchronize()
            ABr = torch.matmul(A, Br)

            if e3 is not None:
                e3.synchronize()
            Cr = torch.matmul(C, r)
        else:
            if e2 is not None:
                e2.synchronize()
            Br = torch.mm(B, r)

            if e1 is not None:
                e1.synchronize()
            ABr = torch.mm(A, Br)

            if e3 is not None:
                e3.synchronize()

            Cr = torch.mm(C, r)

        loss = F.mse_loss(ABr, Cr).item()
        assert loss < threshold, f"Freivalds' algorithm failed with loss {loss}"
        return loss
    # if not torch.allclose(ABr, Cr):
    #    ret = F.mse_loss(ABr, Cr).item()
    # return ret

def copy_to_cpu(x_device: torch.Tensor, stream_copy):
    if x_device.is_cuda:
        x_host = torch.empty_like(x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
        e = torch.cuda.Event()
        with torch.cuda.stream(stream_copy):
            x_host.copy_(x_device, non_blocking=True)
            e.record(stream_copy)
            return x_host, e
    else:
        return x_device, None

class SyncVerifyLinear(Module):
    def __init__(self, linear, gpu_stream, cpu_stream):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.clone().to("cpu")
        self.weight_t = self.weight.t() 
        self.bias = self.linear.bias
        if self.linear.bias is not None:
            self.bias_cpu = linear.bias.clone().to("cpu")
        else:
            self.bias_cpu = None

        self.gpu_stream = gpu_stream
        self.cpu_stream = cpu_stream
            
        self.ctx = []
        self.verify_event = torch.cuda.Event(blocking=True)
        self.compute_event = torch.cuda.Event(blocking=True)

    def forward(self, input):
        t1 = time.time()
        out = F.linear(input, self.linear.weight, bias=None)
        tup = (input.shape, self.linear.weight.shape, out.shape)
        if tup not in all_matmul.keys():
            all_matmul[tup] = 1
        else:
            all_matmul[tup] += 1

        t2 = time.time()
        # print("Linear ", t2-t1)
        input_cpu, e = copy_to_cpu(input, self.cpu_stream)
        self.gpu_stream.synchronize()
        t3 = time.time()
        # print("Copy input ", t3-t2)
        out_cpu, e = copy_to_cpu(out, self.cpu_stream)
        t4 = time.time()
        self.cpu_stream.synchronize()
        # print("Copy output ", t4-t3)
        loss, _ = self.verify_forward(input_cpu, out_cpu)
        self.cpu_stream.synchronize()
        t5 = time.time()
        # print("Verify output", t5-t4)
        return self.add_bias(out)

    def add_bias(self, input):
        if self.linear.bias is not None:
            with torch.cuda.stream(self.gpu_stream):
                return input + self.linear.bias
        else:
            return input

    def add_bias_cpu(self, input):
        if self.linear.bias is not None:
            with torch.cuda.stream(self.cpu_stream):
                return input + self.bias_cpu
        else:
            return input

    def verify_forward(self, input_from_gpu, output_from_gpu):
        assert not input_from_gpu.is_cuda
        assert not output_from_gpu.is_cuda
        with torch.cuda.stream(self.cpu_stream):
            loss = freivalds_algorithm(input_from_gpu, self.weight_t, output_from_gpu)
            self.verify_event.record(self.cpu_stream)
            return loss, None 

    def verify_forward_finegrain(self, input_from_gpu, output_from_gpu):
        self.compute_event.synchronize()
        with torch.cuda.stream(self.cpu_stream):
            input_cpu, e1 = copy_to_cpu(input_from_gpu, self.cpu_stream)
            output_cpu, e2 = copy_to_cpu(output_from_gpu, self.cpu_stream)
            if e1 is not None:
                e1.synchronize()
            if e2 is not None:
                e2.synchronize()
            self.cpu_stream.synchronize()
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu)
            self.verify_event.record(self.cpu_stream)
            return loss, output_cpu

class VerifyLinear:
    def __init__(self, linear, st_cpu, st_gpu, noise=None):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.clone().to("cpu")
        self.weight_t = self.weight.t() 
        self.linear = linear
        if self.linear.bias is not None:
            self.bias_cpu = linear.bias.clone().to("cpu")
            
        self.ctx = []
        self.cpu_stream = st_cpu
        self.gpu_stream = st_gpu 
        self.verify_event = torch.cuda.Event(blocking=True)
        self.compute_event = torch.cuda.Event(blocking=True)
        self.noise = noise

    def forward(self, input):
        with torch.cuda.stream(self.gpu_stream):
            out = F.linear(input, self.linear.weight, bias=None)
            g_logger.debug(f"input mean={input.mean()}, std={input.std()}")
            g_logger.debug(f"weight mean={self.linear.weight.mean()}, std={self.linear.weight.std()}")
            g_logger.debug(f"out mean={out.mean()}, std={out.std()}")
            out = add_noise(out, self.noise)
            self.compute_event.record(self.gpu_stream)
            return out

    def add_bias(self, input):
        if self.linear.bias is not None:
            with torch.cuda.stream(self.gpu_stream):
                return input + self.linear.bias
        else:
            return input

    def add_bias_cpu(self, input):
        if self.linear.bias is not None:
            with torch.cuda.stream(self.cpu_stream):
                return input + self.bias_cpu
        else:
            return input

    def verify_forward(self, input_from_gpu, output_from_gpu):
        assert not input_from_gpu.is_cuda
        assert not output_from_gpu.is_cuda
        with torch.cuda.stream(self.cpu_stream):
            loss = freivalds_algorithm(input_from_gpu, self.weight_t, output_from_gpu)
            self.verify_event.record(self.cpu_stream)
            return loss, None 

    def verify_forward_finegrain(self, input_from_gpu, output_from_gpu):
        self.compute_event.synchronize()
        with torch.cuda.stream(self.cpu_stream):
            input_cpu, e1 = copy_to_cpu(input_from_gpu, self.cpu_stream)
            output_cpu, e2 = copy_to_cpu(output_from_gpu, self.cpu_stream)
            if e1 is not None:
                e1.synchronize()
            if e2 is not None:
                e2.synchronize()
            self.cpu_stream.synchronize()
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu)
            self.verify_event.record(self.cpu_stream)
            return loss, output_cpu

class SparseLinearAlgo(enum.Enum):
    MASK = 1
    TORCH_SPARSE = 2
    FA_SLIDING_WINDOW = 3

class SparseLinear(Linear):

    def __init__(self, in_features, out_features, bias=True, sparse_algo=SparseLinearAlgo.MASK, sparse_mask=None):
        super().__init__(in_features, out_features, bias)
        self.sparse_mask = sparse_mask
        self.sparse_algo = sparse_algo
        if sparse_algo == SparseLinearAlgo.MASK:
            self.weight = self.weight * self.sparse_mask
        elif sparse_algo == SparseLinearAlgo.TORCH_SPARSE:
            self.weight = self.weight.to_sparse_bsc(blocksize=(16, 4), dense_dim=0)
        elif sparse_algo == SparseLinearAlgo.FA_SLIDING_WINDOW:
            assert False, "Not implemented FA"
        else:
            assert False, "Invalid sparse algo"

    def forward(self, input):
        if self.sparse_mask is not None:
            weight = self.weight * self.sparse_mask
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)


def sliding_window_mask(seq_len, window, device="cpu"):
    # indices [L, L]
    arange = torch.arange(seq_len, device=device)
    diff = arange[:, None] - arange[None, :] 
    # allow positions 0..window (causal + window)
    mask = torch.where((diff >= 0) & (diff <= window), 1.0, 0.0)
    return mask

def test_sparse_linear():
    seqlen = 10
    hidden = 5
    window = 3
    mask = sliding_window_mask(seqlen, window)
    linear = SparseLinear(hidden, seqlen, sparse_mask=mask)
    print(linear)

def test_verify(batch, hidden, inter):
    linear = Linear(hidden, inter, bias=False).to("cuda")
    cuda_stream = torch.cuda.Stream()
    cpu_stream = torch.cuda.Stream()
    verify_linear = VerifyLinear(linear, cuda_stream, cpu_stream)

    x = torch.randn(batch, hidden, device="cuda", requires_grad=False)
    y = verify_linear.forward(x)

    #cpu_stream.wait_event(verify_linear.compute_event)
    print("Output from GPU: ", y)
    loss, output_cpu = verify_linear.verify_forward(x, y)
    # cpu_stream.wait_event(verify_linear.verify_event)
    g_logger.info(f"All events issued: {cpu_stream.query()=}, {cuda_stream.query()=}")

    print(f"Loss: {loss}")
    pass

if __name__ == "__main__":
    test_sparse_linear()
    #test_verify(32, 64, 128)