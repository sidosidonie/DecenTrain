import sys
import os
from torch.autograd import Function
from torch.nn import Linear, Module, Parameter
from torch.nn import functional as F, init
from verified_training.verification import time_profile, freivalds_algorithm
import time
import math
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import Tensor
from verified_training.utils.log_utils import g_logger

def copy_to_cpu(x_device: torch.Tensor, stream_copy):
    if x_device.is_cuda:
        x_host = torch.empty_like(x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
        e = torch.cuda.Event(blocking=True)
        with torch.cuda.stream(stream_copy):
            x_host.copy_(x_device, non_blocking=True)
            e.record(stream_copy)
            return x_host, e
    else:
        return x_device, None

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
    def __init__(self, linear: LinearWithMM, st_cpu, st_gpu):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.clone().to("cpu")
        self.weight_t = self.weight.t() 
        self.linear = linear
        if self.linear.bias:
            self.bias_cpu = linear.bias.clone().to("cpu")
            
        self.ctx = []
        self.cpu_stream = st_cpu
        self.gpu_stream = st_gpu 
        self.verify_event = torch.cuda.Event(blocking=True)
        self.compute_event = torch.cuda.Event(blocking=True)

    def forward(self, input):
        with torch.cuda.stream(self.gpu_stream):
            out = self.linear(input)
            self.compute_event.record(self.gpu_stream)
            return out

    def add_bias(self, input):
        if self.linear.bias:
            with torch.cuda.stream(self.gpu_stream):
                return input + self.linear.bias
        else:
            return input

    def add_bias_cpu(self, input):
        if self.linear.bias:
            with torch.cuda.stream(self.cpu_stream):
                return input + self.bias_cpu
        else:
            return input

    def verify_forward_mm(self, input_from_gpu, output_from_gpu):
        self.compute_event.synchronize()
        with torch.cuda.stream(self.cpu_stream):
            input_cpu, e1 = copy_to_cpu(input_from_gpu, self.cpu_stream)
            # stack input
            self.ctx.append(input_cpu)
            output_cpu, e2 = copy_to_cpu(output_from_gpu, self.cpu_stream)
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu, self.cpu_stream, e1, None, e2)
            self.verify_event.record(self.cpu_stream)
            return loss, output_cpu

    def verify_forward(self, input_from_gpu, output_from_gpu):
        self.compute_event.synchronize()
        with torch.cuda.stream(self.cpu_stream):
            input_cpu, e1 = copy_to_cpu(input_from_gpu, self.cpu_stream)
            output_cpu, e2 = copy_to_cpu(output_from_gpu, self.cpu_stream)
            if e1 is not None:
                e1.synchronize()
            if e2 is not None:
                e2.synchronize()
            loss = freivalds_algorithm(input_cpu, self.weight_t, output_cpu, self.cpu_stream, e1, None, e2)
            self.verify_event.record(self.cpu_stream)
            return loss, output_cpu

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

def test_verify(batch, hidden, inter):
    linear = Linear(hidden, inter, bias=False).to("cuda")
    cuda_stream = torch.cuda.Stream()
    cpu_stream = torch.cuda.Stream()
    verify_linear = VerifyLinear(linear, cuda_stream, cpu_stream)

    x = torch.randn(batch, hidden, device="cuda", requires_grad=False)
    y = verify_linear.forward(x)

    #cpu_stream.wait_event(verify_linear.compute_event)
    print("Output from GPU: ", y)
    loss, output_cpu = verify_linear.verify_forward(x, y)
    # cpu_stream.wait_event(verify_linear.verify_event)
    g_logger.info(f"All events issued: {cpu_stream.query()=}, {cuda_stream.query()=}")

    print(f"Loss: {loss}")
    pass

if __name__ == "__main__":
    test_verify(1024, 4096, 8192)