import torch
from verified_training.mlp_layer import copy_to_cpu, freivalds_algorithm, freivalds_algorithm_linear
from verified_training.utils.profiler import Profiler, Duration
from pprint import pprint
import torch.cuda as cuda


def linear(st, a, b):
    e = cuda.Event(blocking=True, enable_timing=True, interprocess=False)
    with cuda.stream(st):
        c = torch.nn.functional.linear(a, b)
        e.record(st)
        return e, c


def verify(st, a, b ,c):
    e = cuda.Event(blocking=True, enable_timing=True, interprocess=False)
    with cuda.stream(st):
        a_cpu = copy_to_cpu(a, st)
        b_cpu = copy_to_cpu(b, st)
        c_cpu = copy_to_cpu(c, st)
        e.record(st)
        e.synchronize()
        loss = freivalds_algorithm_linear(a_cpu, b_cpu, c_cpu)
        e.record(st)
        return e, c_cpu


def three_fc(batch, inc, outc):
    data = torch.randn(batch, inc).cuda()
    print(data)
    weight1 = torch.randn(inc, outc).cuda()
    weight1_cpu = copy_to_cpu(weight1)
    weight2 = torch.randn(outc, inc).cuda()
    weight2_cpu = copy_to_cpu(weight2)

    cuda.synchronize()

    stcpu = cuda.Stream()
    stgpu = cuda.Stream()
    
    # Linear1
    e1, out1 = linear(stgpu, data, weight1)
    # Linear1
    e2, out2 = linear(stgpu, out1, weight2)

    # Verify for linear1
    with cuda.stream(stcpu):
        print("Verify 1")
        e1.synchronize()
        ee, out1_cpu = verify(stcpu, data, weight1_cpu, out1)
        print("Verify 2")
        e2.synchronize()
        ee.synchronize()
        verify(stcpu, out1_cpu, weight2_cpu, out2)

    stcpu.synchronize()
    stgpu.synchronize()
    


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

#main()

three_fc(2048*8, 8192, 8192)