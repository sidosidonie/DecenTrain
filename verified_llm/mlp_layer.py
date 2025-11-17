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
from verified_llm.verify_linear import VerifyLinear, copy_to_cpu
from torch.nn import functional as F, init
from torch.nn import Linear, Module, Parameter
from torch.autograd import Function
import torch
import sys
import os
from verified_llm.utils.log_utils import g_logger
from verified_llm.utils.profiler import Profiler, Duration
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

    def __init__(self, origin: LlamaMLP, stream_cpu: torch.cuda.Stream, stream_gpu: torch.cuda.Stream, noise_scale, is_async = False):
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
        self.gate_proj = VerifyLinear(origin.gate_proj, stream_cpu, stream_gpu, noise_scale)
        self.up_proj = VerifyLinear(origin.up_proj, stream_cpu, stream_gpu, noise_scale)
        self.down_proj = VerifyLinear(origin.down_proj, stream_cpu, stream_gpu, noise_scale)

        self.stream_cpu = stream_cpu
        self.stream_gpu = stream_gpu
        self.event_st = torch.cuda.Event(enable_timing=True, blocking=True)
        self.event_ed = torch.cuda.Event(enable_timing=True, blocking=True)

        self.is_async = is_async

    def act(self, x, stream):
        with torch.cuda.stream(stream):
            out = ACT2FN["silu"](x)
            return out

    def mul(self, x, y, stream):
        with torch.cuda.stream(stream):
            out = x * y
            return out

    def forward(self, x_gpu):
        # Issue all GPU kernels
        self.event_st.record(self.stream_cpu)

        gate = self.gate_proj.forward(x_gpu)
        gate_bias = self.gate_proj.add_bias(gate)
        gate_silu = self.act(gate_bias, self.stream_gpu)
        up = self.up_proj.forward(x_gpu)
        up_bias = self.up_proj.add_bias(up)

        x_cpu, e_copy = copy_to_cpu(x_gpu, self.stream_cpu)
        e_copy.synchronize()

        loss_gate, gate_cpu = self.gate_proj.verify_forward_finegrain(x_cpu, gate)
        g_logger.info(f"MLP_gate loss: {loss_gate}")
        gate_cpu = self.gate_proj.add_bias_cpu(gate_cpu)
        gate_silu_cpu = self.act(gate_cpu, self.stream_cpu)

        loss_up, up_cpu = self.up_proj.verify_forward_finegrain(x_cpu, up)
        g_logger.info(f"MLP_up loss: {loss_up}")
        up_cpu = self.up_proj.add_bias_cpu(up_cpu)
        gate_silu_x_up_cpu = self.mul(gate_silu_cpu, up_cpu, self.stream_cpu)

        gate_silu_x_up = self.mul(gate_silu, up_bias, self.stream_gpu)

        down = self.down_proj.forward(gate_silu_x_up)

        loss_down, down_cpu = self.down_proj.verify_forward_finegrain(gate_silu_x_up_cpu, down)
        g_logger.info(f"MLP_down loss: {loss_down}")
        down = self.down_proj.add_bias(down)

        self.down_proj.verify_event.synchronize()
        self.event_ed.record(self.stream_cpu)

        self.event_ed.synchronize()
        self.event_st.synchronize()
        t = self.event_st.elapsed_time(self.event_ed)
        print("Sync Elapsed time: ", t)
        # return down, down_cpu, t
        return down

    def forward_sync(self, x_gpu):
        # Issue all GPU kernels
        self.event_st.record(self.stream_cpu)

        ## gpu computations
        gate = self.gate_proj.forward(x_gpu)
        gate_bias = self.gate_proj.add_bias(gate)
        gate_silu = self.act(gate_bias, self.stream_gpu)
        up = self.up_proj.forward(x_gpu)
        up_bias = self.up_proj.add_bias(up)
        gate_silu_x_up = self.mul(gate_silu, up_bias, self.stream_gpu)
        down = self.down_proj.forward(gate_silu_x_up)
        down_bias = self.down_proj.add_bias(down)

        self.stream_gpu.synchronize()

        x_cpu, e_copy = copy_to_cpu(x_gpu, self.stream_cpu)

        self.stream_cpu.synchronize()
        loss_gate, gate_cpu = self.gate_proj.verify_forward_finegrain(x_cpu, gate)
        g_logger.info(f"MLP_gate loss: {loss_gate}")
        gate_cpu = self.gate_proj.add_bias_cpu(gate_cpu)
        gate_silu_cpu = self.act(gate_cpu, self.stream_cpu)

        self.stream_cpu.synchronize()
        loss_up, up_cpu = self.up_proj.verify_forward_finegrain(x_cpu, up)
        g_logger.info(f"MLP_up loss: {loss_up}")
        up_cpu = self.up_proj.add_bias_cpu(up_cpu)
        gate_silu_x_up_cpu = self.mul(gate_silu_cpu, up_cpu, self.stream_cpu)

        self.stream_cpu.synchronize()
        loss_down, down_cpu = self.down_proj.verify_forward_finegrain(gate_silu_x_up_cpu, down)
        g_logger.info(f"MLP_down loss: {loss_down}")

        self.event_ed.record(self.stream_cpu)
        self.event_ed.synchronize()
        t = self.event_st.elapsed_time(self.event_ed)
        print("Sync Elapsed time: ", t)
        # return down, down_cpu, t
        return down_bias, t

    def forward_async(self, x_gpu):
        self.event_st.record()

        # gate_proj 
        gate = self.gate_proj.forward(x_gpu)
        x_cpu, e_copy = copy_to_cpu(x_gpu, self.stream_cpu)

        ## ===============

        # gate_proj_bias & up_proj || copy_gate && verify_gate && gate_proj_bias
        ### GPU
        gate_bias = self.gate_proj.add_bias(gate)
        up = self.up_proj.forward(x_gpu)
        ### CPU 
        self.gate_proj.compute_event.synchronize()
        gate_cpu, ee = copy_to_cpu(gate, self.stream_cpu)
        ee.synchronize()
        diff_gate = self.gate_proj.verify_forward(x_cpu, gate_cpu)
        g_logger.info(f"gate diff = {diff_gate}")
        gate_bias_cpu = self.gate_proj.add_bias_cpu(gate_cpu)


        ## ===============


        # up_proj_bias && act_silu && mul_gate_up && down_proj 
        # || copy_up && verify_up &&up_proj_bias && act_silu && mul_gate
        ### GPU
        up_bias = self.up_proj.add_bias(up)
        act_silu = self.act(gate_bias, self.stream_gpu)
        mul_gate_up = self.mul(act_silu, up_bias, self.stream_gpu)
        down = self.down_proj.forward(mul_gate_up)
        ### CPU
        self.up_proj.compute_event.synchronize()
        up_cpu, ee = copy_to_cpu(up, self.stream_cpu)
        ee.synchronize()
        diff_up = self.up_proj.verify_forward(x_cpu, up_cpu)
        g_logger.info(f"up diff = {diff_up}")
        up_bias_cpu = self.up_proj.add_bias_cpu(up_cpu)
        act_silu_cpu = self.act(gate_bias_cpu, self.stream_cpu)
        mul_gate_up_cpu = self.mul(act_silu_cpu, up_bias_cpu, self.stream_cpu)
        
        
        ## ===============

        # down_proj_bias || copy_down && verify_down && down_proj_bias
        ### GPU
        down_bias = self.down_proj.add_bias(down)
        ### CPU
        self.down_proj.compute_event.synchronize()
        down_cpu, ee = copy_to_cpu(down, self.stream_cpu)
        ee.synchronize()
        diff_down = self.down_proj.verify_forward(mul_gate_up_cpu, down_cpu)
        g_logger.info(f"down diff = {diff_down}")
        down_bias_cpu = self.down_proj.add_bias_cpu(down_cpu)

        self.stream_gpu.synchronize()

        self.event_ed.record()
        self.event_ed.synchronize()
        t = self.event_st.elapsed_time(self.event_ed)
        g_logger.info(f"Async Elapsed time: {t}")

        return down_bias, t
        

    def forward_origin(self, x):
        self.event_st.record()
        gate = self.gate_proj.forward(x)
        gate_bias = self.gate_proj.add_bias(gate)
        act_silu = self.act(gate_bias, self.stream_gpu)
        up = self.up_proj.forward(x)
        up_bias = self.up_proj.add_bias(up)
        mul = gate_bias + up_bias
        down = self.down_proj.forward(mul)
        down_bias = self.down_proj.add_bias(down)
        self.event_ed.record()
        self.event_ed.synchronize()
        self.event_st.synchronize()
        t = self.event_st.elapsed_time(self.event_ed)
        return down_bias, t



def test_mlp_async(batch):
    # Create streams
    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    config = LlamaConfig("meta-llama/Llama-3.2-1B-Instruct")
    origin_mlp = LlamaMLP(config).to("cuda")
    mlp = LlamaMLPVerify(origin_mlp, copy_stream, compute_stream, 0, True)

    x = torch.randn(batch, config.hidden_size, device="cuda", requires_grad=False)

    e_st = torch.cuda.Event(enable_timing=True)
    e_ed = torch.cuda.Event(enable_timing=True)

    origin_mlp(x)

    e_st.record()
    for i in range(10):
        origin_mlp(x)
        
    e_ed.record()
    e_ed.synchronize()
    origin_time = e_st.elapsed_time(e_ed)
    print("Origin time: ", origin_time/10.)

    async_time = 0
    sync_time = 0
    origin_time = 0
    for i in range(13):
        y_v, ta = mlp.forward_async(x)
        y_v2, ts = mlp.forward_sync(x)
        y_v3, to = mlp.forward_origin(x)
        assert torch.allclose(y_v2, y_v)
        if i >= 3:
            async_time += ta
            sync_time += ts
            origin_time += to
    
    print("ASync time: ", async_time/10.)
    print("Sync time: ", sync_time/10.)
    print("Origin time: ", origin_time/10.)

if __name__ == "__main__":
    test_mlp_async(2048*4)