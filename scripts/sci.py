import numpy as np
from scipy.linalg.blas import sgemm
import time

N = 2048
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Warm-up
sgemm(1.0, A, B)

# Run 100 rounds
times = []
for _ in range(100):
    start = time.time()
    sgemm(1.0, A, B)
    end = time.time()
    times.append((end - start) * 1000)  # ms

print(f"CBLAS CPU average latency over 100 runs: {np.mean(times):.2f} ms")

