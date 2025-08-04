import torch
from torch.nn import Linear
from verified_llm.verification import freivalds_algorithm
from verified_llm.utils.profiler import Profiler
from torch.optim import AdamW
from pprint import pprint
import seaborn

torch.set_num_threads(32)
iter = 50

def eval_linear(batch, inf, of, dev, prof : Profiler):
    mod = Linear(inf, of, bias=False)
    mod.to(dev)

    mod = mod.train()
    optimizer = AdamW(mod.parameters(), lr=1e-5, weight_decay=0.01)

    block_info = f"{batch}-{inf}-{of}-{dev}"

    f = prof.add_time_span(block_info)
    inp = torch.randn((batch, inf)).to(dev)
    target = torch.randn((batch, of)).to(dev)
    for i in range(iter):
        f.new_iter()
        torch.cuda.synchronize()
        f.record("forward")
        out = mod(inp)
        torch.cuda.synchronize()
        f.record("loss")
        grad = out - target
        grad = grad.sum()
        torch.cuda.synchronize()
        f.record("backward-st")
        grad.backward()
        torch.cuda.synchronize()
        f.record("backward-ed")

        f.record("step-st")
        optimizer.step()
        torch.cuda.synchronize()
        f.record("step-ed")

    pprint(prof.dur_dict())

def eval_copy_verify(batch, inf, of, prof : Profiler):
    in_size = (batch , inf)
    out_size = (batch , of)
    weight_size = (of , inf)
    input = torch.randn(in_size, device=torch.device("cuda"))
    weight = torch.randn(weight_size, device=torch.device("cuda"))
    #output = torch.randn(out_size, device=torch.device("cuda"))
    output = torch.nn.functional.linear(input, weight)
    w_t = weight.t()
    output_t = output.t()

    block_info = f"{batch}-{inf}-{of}-copy"
    p = prof.add_time_span(block_info)
    w_t_cpu = copy_gpu_to_cpu(w_t)
    output_t_cpu = copy_gpu_to_cpu(output_t)

    for i in range(iter):
        p.new_iter()
        p.record("st")
        output_cpu = copy_gpu_to_cpu(output)
        p.record("copy_out")
        input_cpu = copy_gpu_to_cpu(input)
        p.record("copy_grad")
        weight_cpu = copy_gpu_to_cpu(weight)
        p.record("copy_delta_weight")

        ### start verify
        p.record("verify-F-out-st")
        freivalds_algorithm(input_cpu, w_t_cpu, output_cpu)
        p.record("verify-F-out-ed")

        p.record("verify-B-grad-st")
        freivalds_algorithm(output_cpu, weight_cpu, input_cpu)
        p.record("verify-B-grad-ed")

        p.record("verify-B-deltaw-st")
        freivalds_algorithm(output_t_cpu, input_cpu, weight_cpu)
        p.record("verify-B-deltaw-ed")

    pprint(prof.dur_dict())


def copy_gpu_to_cpu(x_device):
    #x_host = torch.empty_like(x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
    #xx = x_device.clone()
    ret = x_device.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    return ret

def eval_forward(batch, inf, of, perf : dict):
    cpu = f"{batch}-{inf}-{of}-cpu"
    gpu = f"{batch}-{inf}-{of}-cuda"
    copy_ver = f"{batch}-{inf}-{of}-copy"

    cpu_time = + perf[copy_ver]["verify-F-out-st-verify-F-out-ed"]
    gpu_time = perf[gpu]["forward-loss"]

    cpu_times = {
        "copy_out" :  perf[copy_ver]["st-copy_out"] ,
        "verify_out" : perf[copy_ver]["verify-F-out-st-verify-F-out-ed"]
    }
    cpu_times["total"] = sum(cpu_times.values())
    gpu_times = {
        "forward" : gpu_time
    }
    gpu_times["total"] = sum(gpu_times.values())
    ret = {
        "cpu" : cpu_times,
        "gpu" : gpu_times,
        "meta" : perf
    }
    return ret

def bandwidth_est(t, shape):
    pass

def eval_backward(batch, inf, of, perf : dict):
    cpu = f"{batch}-{inf}-{of}-cpu"
    gpu = f"{batch}-{inf}-{of}-cuda"
    copy_ver = f"{batch}-{inf}-{of}-copy"
    cpu_times = {
        "update" : perf[cpu]["step-st-step-ed"],
        "verify_grad": perf[copy_ver]["verify-B-grad-st-verify-B-grad-ed"], 
        "verify_delta": perf[copy_ver]["verify-B-deltaw-st-verify-B-deltaw-ed"],
        "cp_delta_w": perf[copy_ver]["copy_grad-copy_delta_weight"], 
        "cp_grad": perf[copy_ver]["copy_out-copy_grad"]
    }
    cpu_times["total"] = sum(cpu_times.values())
    gpu_times = {
        "backward" : perf[gpu]["backward-st-backward-ed"],
        "update": perf[gpu]["step-st-step-ed"]
    }
    gpu_times["total"] = sum(gpu_times.values())

    ret = {
        "cpu" : cpu_times,
        "gpu" : gpu_times,
        "meta" : perf
    }
    return ret

def perf(seq_len, d1, d2):
    prof = Profiler("x.log", skip=3)
    eval_linear(seq_len, d1, d2, torch.device("cpu"), prof)
    eval_linear(seq_len, d1, d2, torch.device("cuda"), prof)
    eval_copy_verify(seq_len, d1, d2, prof)

    return prof.dur_dict()
    
import json

def to_percent(d : dict):
    s = sum(d.values())
    ret = { k : v/s for k, v in d.items()}
    return ret

import matplotlib.pyplot as plt

def to_bar(data : dict, f):
    labels = list(data.keys())
    values = list(data.values())
    total = sum(values)

     #percentages = [value / total * 100 for value in values]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color='skyblue')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{val:.1f}%',
            ha='center',
            va='bottom'
        )

    ax.set_ylabel('Time Percent')
    ax.set_title('Time usage')
    fig.savefig(f)


def full_perf(seq_len = 8192, d1 = 28672, d2 = 8192):
    d = perf(seq_len, d1, d2)
    ff = eval_forward(seq_len, d1, d2, d)
    bb = eval_backward(seq_len, d1, d2, d)
    dump = {
        "forward": ff,
        "backward": bb
    }

    with open(f"pp4-sync/{seq_len}-{d1}-{d2}.json", "w") as fp:
        json.dump(dump, fp, indent=4)

    pprint(dump)

    #to_bar(ff[0], "cpu-for.png")
    #to_bar(bb[0], "cpu-back.png")

if __name__ == "__main__":
    full_perf()
    full_perf(8192, 8192, 8192)
    full_perf(8192, 8192, 28672)
    full_perf(2048, 2048, 2048)