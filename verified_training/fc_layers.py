from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pprint import pprint
from transformers.models.llama.modeling_llama import *
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Callable, Optional, Tuple
from transformers.activations import ACT2FN
from torch import Tensor
import math
import time
from verified_training.verification import time_profile, freivalds_algorithm, freivalds_algorithm_linear
from torch.nn import functional as F, init
from torch.nn import Linear, Module, Parameter
from torch.autograd import Function
import torch
import sys
import os
from verified_training.utils.log_utils import g_logger
from verified_training.utils.profiler import Profiler, Duration

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

torch.set_num_threads(32)

def all_close(a, b):
    return 0
    a = a.to("cpu")
    b = b.to("cpu")
    loss = F.mse_loss(a, b).item()
    if loss > 1:
        exit(-1)
    return loss

def copy_to_cpu(x_device: torch.Tensor, stream_copy=None):
    if x_device.is_cuda:
        if stream_copy:
            with torch.cuda.stream(stream_copy):
                x_host = torch.empty_like(
                    x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
                x_host.copy_(x_device, non_blocking=True)
                return x_host
        else:
            x_host = torch.empty_like(
                x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
            x_host.copy_(x_device, non_blocking=True)
            return x_host
    else:
        return x_device

class LinearWithMM(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight : Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((in_features, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return torch.mm(input, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class VerifyLinear:

    def __init__(self, linear: LinearWithMM, st):
        self.original_module = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.clone().to("cpu")
        self.weight_t = self.weight.t() 
        self.linear = linear
        self.ctx = []
        self.stream = st

    @torch.enable_grad()
    def forward(self, input):
        return self.linear(input)

    def verify_forward_mm(self, input_from_gpu, output_from_gpu):
        # copy output to cpu
        with torch.cuda.stream(self.stream):
            input_cpu = copy_to_cpu(input_from_gpu, self.stream)
            output_cpu = copy_to_cpu(output_from_gpu, self.stream)
            self.stream.synchronize()
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu)
            g_logger.info(f"forward loss {loss}")
            self.ctx.append(input_cpu)
            return output_cpu

    def verify_forward(self, input_from_gpu, output_from_gpu):
        # copy output to cpu
        with torch.cuda.stream(self.stream):
            input_cpu = copy_to_cpu(input_from_gpu)
            output_cpu = copy_to_cpu(output_from_gpu)
            self.stream.synchronize()
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu)
            g_logger.info(f"forward loss {loss}")
            self.ctx.append(input_cpu)
            return output_cpu

    def verify_backward(self, grad_output_gpu, grad_input_gpu, grad_weight_gpu):
        """
        Verify grad_input and grad_weight, and return them
        grad_input = mm(grad_output, weight)
        grad_weight = mm(grad_output, input)
        """
        input_cpu = self.ctx[0]
        weight_cpu = self.weight

        grad_output_cpu = copy_to_cpu(grad_output_gpu, self.stream)
        grad_input_cpu = copy_to_cpu(grad_input_gpu, self.stream)
        grad_weight_cpu = copy_to_cpu(grad_weight_gpu, self.stream)

        self.stream.synchronize()
        loss1 = freivalds_algorithm(grad_output_cpu, weight_cpu, grad_input_cpu)
        loss2 = freivalds_algorithm(grad_output_cpu, input_cpu, grad_weight_cpu)
        g_logger.info(f"backward loss1 {loss1}")
        g_logger.info(f"backward loss2 {loss2}")

        return grad_input_cpu, grad_weight_cpu

def llama_mlp(batch, hidden, inter, input_tensor, gate=None, up=None, down=None, prof=None):

    # Create a Linear layer
    if gate is None:
        gate = Linear(hidden, inter, bias=False).to("cuda")
    if up is None:
        up = Linear(hidden, inter, bias=False).to("cuda")
    if down is None:
        down = Linear(inter, hidden, bias=False).to("cuda")

    stream_copy = torch.cuda.Stream()
    #stream_copy = torch.cuda.default_stream(torch.device("cuda"))
    # Create a VerifyLinear instance
    gate_ver = VerifyLinear(gate, stream_copy)
    up_ver = VerifyLinear(up, stream_copy)
    down_ver = VerifyLinear(down, stream_copy)

    input_tensor_cpu = input_tensor.to("cpu")

    torch.cuda.synchronize()
    prof.record()

    # Forward pass
    act_gate = gate_ver.forward(input_tensor)
    torch.cuda.synchronize()
    act_gate_out = ACT2FN["silu"](act_gate)
    up_proj = up_ver.forward(input_tensor)

    act_gate_cpu = gate_ver.verify_forward(input_tensor_cpu, act_gate)
    act_gate_cpu = ACT2FN["silu"](act_gate_cpu)

    torch.cuda.synchronize()
    up_proj_cpu = up_ver.verify_forward(input_tensor_cpu, up_proj)

    act_up = act_gate_out * up_proj
    down_proj = down_ver.forward(act_up)

    # act * up
    act_up_cpu = act_gate_cpu * up_proj_cpu

    # down
    torch.cuda.synchronize()
    down_proj_cpu = down_ver.verify_forward(act_up_cpu, down_proj) 
    torch.cuda.synchronize()
    prof.record()

    ########### backward 
    #return down_proj, down_proj_cpu
    return down_proj, None


class LlamaMLP(Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        #gate = self.gate_proj(x)
        #up = self.up_proj(x)
        #act = self.act_fn(gate)
        #act_up = act * up
        #down = self.down_proj(act_up)
        #return down

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def llama_mlp_test(batch = 12384, hidden = 8192, inter = 8192):
    prof = Profiler("log")
    p1 = prof.add_time_span("original_mlp")
    p2 = prof.add_time_span("verify_mlp")
    mlp = LlamaMLP(hidden, inter)
    mlp.to("cuda")
    x = torch.randn(batch, hidden, device="cuda", requires_grad=True)

    for _ in range(10):
        g_logger.info("###########################")
        p1.new_iter()
        torch.cuda.synchronize()
        p1.record()
        y = mlp(x)
        torch.cuda.synchronize()
        p1.record()

        p2.new_iter()
        ver_mlp, ver_mlp_cpu = llama_mlp(batch, hidden, inter, x, mlp.gate_proj, mlp.up_proj, mlp.down_proj, p2)

    pprint(prof.dur_dict())
    origin = prof.dur_dict()["original_mlp"]["0-1"]
    verify = prof.dur_dict()["verify_mlp"]["0-1"]
    g_logger.info(f"Overhead: {(verify - origin) / origin}")

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
        #y_cpu = l1(x_cpu)
        y1_gpu = l1(x_gpu)
        y1_gpu.retain_grad()
        #y_cpu = v1(x_cpu, y_gpu)
        y1_cpu = v1.forward(x_cpu, y1_gpu)

        y2_gpu = l2(y1_gpu)
        y2_gpu.retain_grad()
        y2_cpu = v2.forward(y1_cpu, y2_gpu)
        y2_cpu.retain_grad()

        y2_gpu.backward(out_grad, retain_graph=True)

        grad = (out_grad_cpu, y1_gpu.grad, l2.weight.grad)
        g_logger.info(grad)

        grad_in_cpu, grad_w2_cpu = v2.backward(out_grad_cpu, y1_gpu.grad, l2.weight.grad)

        y1_cpu.grad = grad_in_cpu
        v2.weight.grad = grad_w2_cpu

        g_logger.info(y1_cpu.grad)

    #y_cpu.backward(gradient = final_grad, inputs = [x_gpu.grad, linear_gpu.weight.grad])


if __name__ == "__main__":
    #gpu_cpu_modeul()
    llama_mlp_test()


