import torch
import torch.nn.functional as F

x = torch.randn(1024, 1024)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True
) as prof:
    F.softmax(x, dim=1)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


model = torch.nn.Sequential(torch.nn.Softmax(dim=1))
x = torch.randn(32, 1000)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

import time
import torch.nn as nn


start = time.time()
for _ in range(100):
    y = F.softmax(x, dim=1)
torch_cpu_time = time.time() - start
print(f"ATen softmax time: {torch_cpu_time:.4f} sec")


model = nn.Sequential(nn.Softmax(dim=1))
x_mkldnn = x.clone().to_mkldnn()  # Convert to MKLDNN layout
start = time.time()
for _ in range(100):
    y = model(x_mkldnn)
mkldnn_time = time.time() - start
print(f"MKL-DNN softmax time: {mkldnn_time:.4f} sec")
