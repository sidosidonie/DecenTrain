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

    def __init__(self, linear: LinearWithMM, st_cpu, st_gpu=None):
        self.original_module = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.clone().to("cpu")
        self.weight_t = self.weight.t() 
        self.linear = linear
        self.ctx = []
        self.cpu_stream = st_cpu
        if st_gpu is None:
            self.gpu_stream = torch.cuda.default_stream()
        else:
            self.gpu_stream = st_gpu 

    @torch.enable_grad()
    def forward(self, input):
        return self.linear(input)

    def verify_forward_mm(self, input_from_gpu, output_from_gpu):
        # copy output to cpu
        with torch.cuda.stream(self.cpu_stream):
            input_cpu = copy_to_cpu(input_from_gpu, self.cpu_stream)
            output_cpu = copy_to_cpu(output_from_gpu, self.cpu_stream)
            #self.stream.synchronize()
            ee = torch.cuda.Event(blocking=True)
            ee.record()
            ee.synchronize()
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu)
            g_logger.info(f"forward loss {loss}")
            self.ctx.append(input_cpu)
            return output_cpu

    def verify_forward(self, input_from_gpu, output_from_gpu):
        # copy output to cpu
        with torch.cuda.stream(self.cpu_stream):
            output_cpu = copy_to_cpu(output_from_gpu)
            input_cpu = copy_to_cpu(input_from_gpu)
            ee = torch.cuda.Event(blocking=True)
            ee.record()
            ee.synchronize()
            #self.stream.wait_event(ee)
            #self.stream.synchronize()
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

        grad_output_cpu = copy_to_cpu(grad_output_gpu, self.cpu_stream)
        grad_input_cpu = copy_to_cpu(grad_input_gpu, self.cpu_stream)
        grad_weight_cpu = copy_to_cpu(grad_weight_gpu, self.cpu_stream)

        self.cpu_stream.synchronize()
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
        act_gate_cpu = gate_ver.verify_forward(input_tensor_cpu, act_gate)
        act_gate_cpu = ACT2FN["silu"](act_gate_cpu)

    stream_copy.wait_event(e2)
    up_proj_cpu = up_ver.verify_forward(input_tensor_cpu, up_proj)

    act_up = act_gate_out * up_proj
    down_proj = down_ver.forward(act_up)
    e3.record()

    # act * up
    with torch.cuda.stream(stream_copy):
        act_up_cpu = act_gate_cpu * up_proj_cpu

    # down
    stream_copy.wait_event(e3)
    down_proj_cpu = down_ver.verify_forward(act_up_cpu, down_proj) 

    ########### backward 
    #return down_proj, down_proj_cpu
    stream_copy.synchronize()
    torch.cuda.synchronize()
    e_ed.record()
    print("Verify ", e_st.elapsed_time(e_ed))
    return e_st.elapsed_time(e_ed), down_proj, None

class LlamaMLPVerify(torch.nn.Module):

    def __init__(self, origin : LlamaMLP, stream_cpu, stream_gpu):
        self.gate_proj = VerifyLinear(origin.gate_proj, stream_cpu, stream_gpu) 
        self.up_proj = VerifyLinear(origin.up_proj, stream_cpu, stream_gpu) 
        self.down_proj = VerifyLinear(origin.down_proj, stream_cpu, stream_gpu) 

    def forward(self, x):
        pass


def llama_mlp_test(iter = 100, batch = 1024):
    prof = Profiler("log")
    p1 = prof.add_time_span("original_mlp")
    p2 = prof.add_time_span("verify_mlp")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config = LlamaConfig(model_name)
    config.mlp_bias=False
    print(config)

    time.sleep(0.1)
    mlp = LlamaMLP(config)
    mlp.to("cuda")

    x = torch.randn(batch, config.hidden_size, device="cuda", requires_grad=True)
    total_origin = 0 
    total_verify = 0 
    e_st = torch.cuda.Event(enable_timing=True)
    e_ed = torch.cuda.Event(enable_timing=True)

    for i in range(iter):
        g_logger.info("###########################")
        torch.cuda.synchronize()
        e_st.record()
        t, ver_mlp, ver_mlp_cpu = llama_mlp(batch, config.hidden_size, config.intermediate_size, x, mlp.gate_proj, mlp.up_proj, mlp.down_proj, p2)
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
    #origin = prof.dur_dict()["original_mlp"]["0-1"]
    #verify = prof.dur_dict()["verify_mlp"]["0-1"]
    g_logger.info(f"Origin: {origin_time}, Verify: {verify_time} Overhead: {(verify_time - origin_time) / origin_time}")

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
    ll = {
        "batch" : [16, 32, 128, 256, 512, 1024, 2048, 4096, 8192 ,12384],
        "origin" : [],
        "verify" : [],
        "overhead" : []
    }
    #for b in [16, 32, 128, 256, 512, 1024, 2048, 4096]:
    for b in ll["batch"]:
        origin, verify, overhead = llama_mlp_test(iter=100, batch=b)
        ll["origin"].append(origin)
        ll["verify"].append(verify)
        ll["overhead"].append(overhead)

    df = pd.DataFrame(ll)
    print(df)
    print(df.to_markdown())
