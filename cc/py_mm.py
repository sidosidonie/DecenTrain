import torch
import time
import numpy as np
from torch.nn import functional as F, init
from verified_training.verification import freivalds_algorithm


def verify(dev, m, n, k):
    torch.set_num_threads(32)
    device = torch.device(dev)

    # Create matrices
    A = torch.randn(m, k, device=device, dtype=torch.float32)
    B = torch.randn(k, n, device=device, dtype=torch.float32)
    C = torch.mm(A, B)

    times = []
    nn = C.shape[-1]
    r = torch.randn((nn, 10), dtype=torch.float32, device=dev)
    for _ in range(100):
        #freivalds_algorithm(A, B, C)
        start = time.time()
        Br = torch.mm(B, r)
        ABr = torch.mm(A, Br)
        Cr = torch.mm(C, r)
        end = time.time()
        times.append((end - start)* 1000)

    res = np.mean(times)
    print(f"Verify on {dev}: {np.mean(times):.3f} ms")
    return res

def main(dev, m, n, k, use_mat, dtype):
    # Set to CPU
    torch.set_num_threads(32)
    device = torch.device(dev)

    # Create matrices
    A = torch.randn(m, k, device=device, dtype=dtype)
    if use_mat:
        B = torch.randn(n, k, device=device, dtype=dtype)
    else:
        B = torch.randn(k, n, device=device, dtype=dtype)

    # Warm-up
    if use_mat:
        F.linear(A, B)
    else:
        torch.mm(A, B)

    # Time it
    times = []
    for _ in range(100):
        if dev == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        if use_mat:
            C = F.linear(A, B)
        else:
            C = torch.mm(A, B)
        if dev == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # ms

    print(f"Use_linear: {use_mat} {dev}: {np.mean(times):.3f} ms")
    return np.mean(times)

m = 8192
n = 8192
k = 8192 

def tops(m, n, k, t):
    return m*n*k*2/1e9 / t

def mm(m, n, k):
    print(f">>>> {m=}, {n=}, {k=}")
    t2 = main("cuda", m, n, k, True, torch.float16)
    print(f"cuda float16 tops = {tops(m, n, k, t2)}")
    t2 = main("cuda", m, n, k, True, torch.float32)
    print(f"cuda float32 tops = {tops(m, n, k, t2)}")
    t1 = main("cpu", m, n, k, False, torch.float32)
    print(f"cpu float32 tops = {tops(m, n, k, t1)}")
    verify("cpu", m, n, k)
    print(t1/t2)

mm(8192, 8192, 8192)
exit(-1)

mm(8192, 28672, 8192)
mm(8192, 8192, 28672)
