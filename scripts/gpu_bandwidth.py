import numpy as np
import cupy as cp
import time
import torch

def test_gpu_bandwidth_torch(size_in_mb=1024, pin=True, dtype=torch.float32):
    size = size_in_mb * 1024 * 1024 # Convert MB to bytes
    num_elements = size // torch.tensor([], dtype=dtype).element_size()

    print(f"Testing GPU bandwidth with {size_in_mb} MB of data...")

    # Allocate host (CPU) and device (GPU) tensors
    h_tensor = torch.randn(num_elements, dtype=dtype, device='cpu', pin_memory=pin)
    d_tensor = torch.empty(num_elements, dtype=dtype, device='cuda')

    # Host to Device (H2D) transfer
    torch.cuda.synchronize()
    start = time.time()
    d_tensor.copy_(h_tensor)
    torch.cuda.synchronize()
    h2d_time = time.time() - start

    # Device to Host (D2H) transfer
    h_tensor2 = torch.empty(num_elements, dtype=dtype, device='cpu', pin_memory=pin)
    torch.cuda.synchronize()
    start = time.time()
    h_tensor2.copy_(d_tensor)
    torch.cuda.synchronize()
    d2h_time = time.time() - start

    h2d_bandwidth = size_in_mb / 1024. / h2d_time
    d2h_bandwidth = size_in_mb / 1024. / d2h_time

    return h2d_bandwidth, d2h_bandwidth


def test_bandwidth(pin, rep=10):
    h2d_pin_time = 0
    d2h_pin_time = 0
    for i in range(rep):
        h2d, d2h = test_gpu_bandwidth_torch(1024, pin)  # Test with 1 GB
        h2d_pin_time += h2d
        d2h_pin_time += d2h

    h2d = h2d_pin_time / rep
    d2h = d2h_pin_time / rep

    print("==== PinMemory: ", pin)
    print(f"Host to device bandwidth: {h2d:.2f} GB/s")
    print(f"Device to host bandwidth: {d2h:.2f} GB/s")

test_bandwidth(True)
test_bandwidth(False)