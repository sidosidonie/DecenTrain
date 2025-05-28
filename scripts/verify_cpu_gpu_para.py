import torch
from verified_training.layer import copy_to_cpu
from verified_training.utils.profiler import Profiler, Duration
from pprint import pprint

def gpu_only(p : Duration):
    a = torch.rand(4096, 8192).cuda()
    b = torch.rand(8192, 4096).cuda()

    torch.cuda.synchronize()
    p.record()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    p.record()
    return c

def with_copy(p : Duration):
    a = torch.rand(4096, 8192).cuda()
    b = torch.rand(8192, 4096).cuda()
    d = torch.rand(8192, 4096).cuda()

    d_cpy = torch.empty_like(d, device="cpu", pin_memory=True)

    stream = torch.cuda.Stream()
    p.record()
    c = torch.mm(a, b)
    #c_cpu = copy_to_cpu(d)
    with torch.cuda.stream(stream):
        d_cpy.copy_(d, non_blocking=True)
    torch.cuda.synchronize()
    stream.synchronize()
    p.record()
    return c 

def with_sync(p : Duration):
    a = torch.rand(4096, 8192).cuda()
    b = torch.rand(8192, 4096).cuda()
    d = torch.rand(8192, 4096).cuda()

    torch.cuda.synchronize()
    p.record()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    p.record()
    c_cpu = copy_to_cpu(d)
    torch.cuda.synchronize()
    p.record()
    return c, c_cpu


def main():
    profiler = Profiler("log")
    gpu = profiler.add_time_span("gpu")
    copy = profiler.add_time_span("copy")
    sync = profiler.add_time_span("sync")
    for _ in range(10):
        gpu.new_iter()
        copy.new_iter()
        sync.new_iter()
        gpu_only(gpu)
        with_copy(copy)
        with_sync(sync)

    pprint(profiler.dur_dict())

main()