from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pprint import pprint
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Callable, Optional, Tuple
from transformers.activations import ACT2FN
from torch import Tensor
import math
import time
from verified_training.verification import time_profile, freivalds_algorithm, freivalds_algorithm_linear
from verified_training.verify_linear import VerifyLinear, copy_to_cpu
from torch.nn import functional as F, init
from torch.nn import Linear, Module, Parameter
from torch.autograd import Function
import torch
import sys
import os
from verified_training.utils.log_utils import g_logger
from verified_training.utils.profiler import Profiler, Duration
from torch import nn

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

torch.set_num_threads(32)


def llama_mlp(batch, hidden, inter, input_tensor, gate=None, up=None, down=None, prof=None):

    # Create a Linear layer
    if gate is None:
        gate = Linear(hidden, inter, bias=False).to("cuda")
    if up is None:
        up = Linear(hidden, inter, bias=False).to("cuda")
    if down is None:
        down = Linear(inter, hidden, bias=False).to("cuda")

    stream_copy = torch.cuda.Stream()
    stream_gpu = torch.cuda.Stream()
    # stream_copy = torch.cuda.default_stream(torch.device("cuda"))
    # Create a VerifyLinear instance
    gate_ver = VerifyLinear(gate, stream_copy, stream_gpu)
    up_ver = VerifyLinear(up, stream_copy, stream_gpu)
    down_ver = VerifyLinear(down, stream_copy, stream_gpu)

    input_tensor_cpu = input_tensor.to("cpu")

    e_st = torch.cuda.Event(enable_timing=True)
    e_ed = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    e_st.record()
    stream_copy.synchronize()

    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)
    # Forward pass
    act_gate = gate_ver.forward(input_tensor)
    e1.record()
    act_gate_out = ACT2FN["silu"](act_gate)
    up_proj = up_ver.forward(input_tensor)
    e2.record()

    stream_copy.wait_event(e1)
    with torch.cuda.stream(stream_copy):
        loss, act_gate_cpu = gate_ver.verify_forward(
            input_tensor_cpu, act_gate)
        act_gate_cpu = ACT2FN["silu"](act_gate_cpu)

    stream_copy.wait_event(e2)
    loss_up, up_proj_cpu = up_ver.verify_forward(input_tensor_cpu, up_proj)

    act_up = act_gate_out * up_proj
    down_proj = down_ver.forward(act_up)
    e3.record()

    # act * up
    with torch.cuda.stream(stream_copy):
        act_up_cpu = act_gate_cpu * up_proj_cpu

    # down
    stream_copy.wait_event(e3)
    loss_down, down_proj_cpu = down_ver.verify_forward(act_up_cpu, down_proj)

    # backward
    # return down_proj, down_proj_cpu
    stream_copy.synchronize()
    stream_gpu.synchronize()
    e_ed.record()
    print("Verify ", e_st.elapsed_time(e_ed))
    return e_st.elapsed_time(e_ed), down_proj, None


class LlamaMLPVerify(torch.nn.Module):

    def __init__(self, origin: LlamaMLP, stream_cpu: torch.cuda.Stream, stream_gpu: torch.cuda.Stream):
        """
        gate = gate_proj(x) || up = up_proj(x)
        gate_cpu = copy_to_cpu(gate)
        verify(gate_cpu, x)
        up_cpu = copy_to_cpu(up)
        verify(up_cpu, x)

        gate_silu = silu(gate) * up
        gate_silu_cpu =silu(gate_cpu) * up_cpu

        down = down_proj(gate_silu)
        down_cpu = copy_to_cpu(down)

        verify(down_cpu, gate_silu_cpu)

        return down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        super().__init__()
        self.gate_proj = VerifyLinear(origin.gate_proj, stream_cpu, stream_gpu)
        self.up_proj = VerifyLinear(origin.up_proj, stream_cpu, stream_gpu)
        self.down_proj = VerifyLinear(origin.down_proj, stream_cpu, stream_gpu)
        self.stream_cpu = stream_cpu
        self.stream_gpu = stream_gpu
        self.event_st = torch.cuda.Event(enable_timing=True, blocking=True)
        self.event_ed = torch.cuda.Event(enable_timing=True, blocking=True)

    def act(self, x, stream):
        with torch.cuda.stream(stream):
            out = ACT2FN["silu"](x)
            return out

    def mul(self, x, y, stream):
        with torch.cuda.stream(stream):
            out = x * y
            return out

    def forward(self, x_gpu):
        g_logger.info("==== Verify MLP ====")
        # Issue all GPU kernels
        g_logger.info("==== GPU Gate, act and up ====")
        self.event_st.record(self.stream_cpu)
        gate = self.gate_proj.forward(x_gpu)
        gate_silu = self.act(gate, self.stream_gpu)
        up = self.up_proj.forward(x_gpu)

        g_logger.info(">>> Verify gate projection <<<")
        x_cpu, e_copy = copy_to_cpu(x_gpu, self.stream_cpu)
        e_copy.synchronize()
        loss_gate, gate_cpu = self.gate_proj.verify_forward(x_cpu, gate)
        gate_silu_cpu = self.act(gate_cpu, self.stream_cpu)

        g_logger.info(">>> Verify up projection <<<")
        loss_up, up_cpu = self.up_proj.verify_forward(x_cpu, up)
        gate_silu_x_up_cpu = self.mul(gate_silu_cpu, up_cpu, self.stream_cpu)

        g_logger.info("=== GPU Mul, Down projection ===")
        gate_silu_x_up = self.mul(gate_silu, up, self.stream_gpu)
        down = self.down_proj.forward(gate_silu_x_up)

        g_logger.info(">>> Verify down projection <<<")
        loss_down, down_cpu = self.down_proj.verify_forward(
            gate_silu_x_up_cpu, down)

        self.down_proj.verify_event.synchronize()
        self.event_ed.record(self.stream_cpu)

        t = self.event_st.elapsed_time(self.event_ed)
        g_logger.info(
            f"Loss gate: {loss_gate}, Loss up: {loss_up}, Loss down: {loss_down}")
        # return down, down_cpu, t
        g_logger.info("==== Verify MLP Done ====")
        return down


def llama_mlp_test(iter=100, batch=1024):
    prof = Profiler("log")
    p1 = prof.add_time_span("original_mlp")
    p2 = prof.add_time_span("verify_mlp")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config = LlamaConfig(model_name)
    config.mlp_bias = False
    print(config)

    time.sleep(0.1)
    mlp = LlamaMLP(config)
    mlp.to("cuda")

    x = torch.randn(batch, config.hidden_size,
                    device="cuda", requires_grad=True)
    total_origin = 0
    total_verify = 0
    e_st = torch.cuda.Event(enable_timing=True)
    e_ed = torch.cuda.Event(enable_timing=True)

    for i in range(iter):
        g_logger.info("###########################")
        torch.cuda.synchronize()
        e_st.record()
        t, ver_mlp, ver_mlp_cpu = llama_mlp(
            batch, config.hidden_size, config.intermediate_size, x, mlp.gate_proj, mlp.up_proj, mlp.down_proj, p2)
        torch.cuda.synchronize()
        e_ed.record()
        if i < 5:
            continue
        else:
            total_verify += t

    verify_time = total_verify / (iter-5)
    print("Verified time: ", verify_time)

    for i in range(iter):
        g_logger.info("###########################")

        torch.cuda.synchronize()
        e_st.record()
        y = mlp(x)
        torch.cuda.synchronize()
        e_ed.record()

        time.sleep(0.1)

        t = e_st.elapsed_time(e_ed)
        if i >= 5:
            total_origin += t

    origin_time = total_origin / (iter-5)
    print("Origin time: ", origin_time)
    # origin = prof.dur_dict()["original_mlp"]["0-1"]
    # verify = prof.dur_dict()["verify_mlp"]["0-1"]
    g_logger.info(
        f"Origin: {origin_time}, Verify: {verify_time} Overhead: {(verify_time - origin_time) / origin_time}")

    return (origin_time, verify_time, (verify_time - origin_time) / origin_time)


def gpu_cpu_modeul(batch=4, in_feat=2, out_feat=3):
    torch.cuda.init()
    l1 = LinearWithMM(in_features=in_feat, out_features=out_feat, bias=None)
    l2 = LinearWithMM(in_features=out_feat, out_features=in_feat, bias=None)
    v1 = VerifyLinear(l1)
    v2 = VerifyLinear(l2)

    l1.to("cuda")
    l1.train
    l2.to("cuda")
    l2.train

    # make input
    x_cpu = torch.randn(batch, in_feat, requires_grad=True)
    x_gpu = x_cpu.to("cuda")
    x_gpu.requires_grad_(True)

    out_grad = torch.randn(batch, in_feat, device="cuda", requires_grad=True)
    out_grad_cpu = out_grad.to("cpu")
    out_grad_cpu.requires_grad_(True)
    out_grad_cpu.retain_grad()

    with torch.enable_grad():
        # y_cpu = l1(x_cpu)
        y1_gpu = l1(x_gpu)
        y1_gpu.retain_grad()
        # y_cpu = v1(x_cpu, y_gpu)
        y1_cpu = v1.forward(x_cpu, y1_gpu)

        y2_gpu = l2(y1_gpu)
        y2_gpu.retain_grad()
        y2_cpu = v2.forward(y1_cpu, y2_gpu)
        y2_cpu.retain_grad()

        y2_gpu.backward(out_grad, retain_graph=True)

        grad = (out_grad_cpu, y1_gpu.grad, l2.weight.grad)
        g_logger.info(grad)

        grad_in_cpu, grad_w2_cpu = v2.backward(
            out_grad_cpu, y1_gpu.grad, l2.weight.grad)

        y1_cpu.grad = grad_in_cpu
        v2.weight.grad = grad_w2_cpu

        g_logger.info(y1_cpu.grad)

    # y_cpu.backward(gradient = final_grad, inputs = [x_gpu.grad, linear_gpu.weight.grad])


def test_mlp(itern, batch, seq_len, config: LlamaConfig):
    mlp = LlamaMLP(config)
    mlp.to("cuda")
    vmlp = LlamaMLPVerify(mlp, torch.cuda.Stream(), torch.cuda.Stream())

    x = torch.randn(batch * seq_len, config.hidden_size,
                    device="cuda", requires_grad=False)
    x_cpu = x.clone().to("cpu")

    total_verify = 0
    total_origin = 0
    for i in range(itern):
        vmlp.stream_cpu.synchronize()
        vmlp.stream_gpu.synchronize()

        out, out_cpu, verify_time = vmlp.forward(x, x_cpu)

        st_ori = torch.cuda.Event(enable_timing=True, blocking=True)
        ed_ori = torch.cuda.Event(enable_timing=True, blocking=True)

        st_ori.record()
        out_origin = mlp(x)
        ed_ori.record()
        torch.cuda.synchronize()
        t_origin = st_ori.elapsed_time(ed_ori)
        g_logger.info(
            f"Iteration {i+1}/{itern}, Verify Time: {verify_time} ms, Origin Time taken: {t_origin} ms")

        diff = F.mse_loss(out, out_origin)
        g_logger.info(f"Iteration {i+1}/{itern}, MSE Loss: {diff.item()}")


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config = LlamaConfig(model_name)
    config.mlp_bias = False
    test_mlp(10, batch=1, seq_len=2048, config=config)
    # ll = {
    #    "batch" : [16, 32, 128, 256, 512, 1024, 2048, 4096, 8192 ,12384],
    #    "origin" : [],
    #    "verify" : [],
    #    "overhead" : []
    # }
    # for b in [16, 32, 128, 256, 512, 1024, 2048, 4096]:
    # for b in ll["batch"]:
    #    origin, verify, overhead = llama_mlp_test(iter=100, batch=b)
    #    ll["origin"].append(origin)
    #    ll["verify"].append(verify)
    #    ll["overhead"].append(overhead)

    # df = pd.DataFrame(ll)
    # print(df)
    # print(df.to_markdown())
