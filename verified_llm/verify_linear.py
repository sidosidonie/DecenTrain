from torch.nn import Linear, Module, Parameter
from torch.nn import functional as F, init
import math
import pandas as pd
import torch
from torch import Tensor
from verified_llm.utils.log_utils import g_logger

class GlobRandomVec:

    def __init__(self):
        self.vec = {}

    def get_key(self, n, k, dtype):
        return f"{n}-{k}-{dtype}"

    def get_or_create_vec(self, n, k, dtype):
        key = self.get_key(n, k, dtype)
        if key in self.vec.keys():
            return self.vec[key]
        else:
            r_vec = torch.randn((n, k), dtype=dtype, device="cpu")
            print("Create rand vec " + key)
            self.vec[key] = r_vec
            return self.vec[key]

glob_random_vec = GlobRandomVec()

def freivalds_algorithm_linear(A, B, C, k=10):
    n = C.shape[-1]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
    # print(f"{r.shape=}")
    # r = torch.randn((n, k), dtype=torch.float16, device=A.device)
    # Br = F.linear(B, r)
    # ABr = F.linear(A, Br)
    # Cr = F.linear(C, r)
    Br = torch.mm(B, r)
    ABr = torch.mm(A, Br)
    Cr = torch.mm(C, r)

    ret = F.mse_loss(ABr, Cr).item()
    if ret > 1:
        print(f"freivalds_algorithm: {ret=}")
        print(f"Cr = {Cr}")
        print(f"ABr = {ABr}")
        exit(-1)
    return ret

def verify_attn_weight_freivalds(q, k, attn_weight, stream, k_freivalds=10):
    """
    q: [batch, head, seq_len, dim]
    k: [batch, head, dim, seq_len]
    attn_weight: [batch, head, seq_len, seq_len]
    """
    batch, head, seq_len_q, dim = q.shape
    _, _, seq_len_k, _ = k.shape

    # Transpose k for matmul: [batch, head, dim, seq_len_k]

    losses = []
    for b in range(batch):
        for h in range(head):
            # q[b, h]: [dim, seq_len_q], k_t[b, h]: [dim, seq_len_k], attn_weight[b, h]: [seq_len_q, seq_len_k]
            loss = freivalds_algorithm(
                q[b, h], k[b, h], attn_weight[b, h], stream, k=k_freivalds
            )
            losses.append(loss)
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return avg_loss

def freivalds_batch_matmul(A, B, C, k=10):
    g_logger.info(f"freivalds_batch_matmul: {A.shape=}, {B.shape=}, {C.shape=}")
    assert A.device == B.device == C.device, "All tensors must be on the same device"
    n = C.shape[-1]
    r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
    Br = torch.matmul(B, r)
    ABr = torch.matmul(A, Br)
    Cr = torch.matmul(C, r)
    loss = F.mse_loss(ABr, Cr).item()
    return loss

def freivalds_algorithm(A, B, C, stream, e1=None, e2=None, e3=None, k=10):
    with torch.cuda.stream(stream):
        n = C.shape[-1]
        r = glob_random_vec.get_or_create_vec(n, k, A.dtype)
        # r = torch.randn((n, k), dtype=torch.float16, device=A.device)
        if len(A.shape) > 2:
            # change A from (n, n, k) to (n*n, k)
            # A = A.reshape(-1, A.shape[-1])
            # B = B.reshape(-1, B.shape[-1])
            # C = C.reshape(-1, C.shape[-1])
            if e2 is not None:
                e2.synchronize()
            Br = torch.matmul(B, r)

            if e1 is not None:
                e1.synchronize()
            ABr = torch.matmul(A, Br)

            if e3 is not None:
                e3.synchronize()
            Cr = torch.matmul(C, r)
        else:
            if e2 is not None:
                e2.synchronize()
            Br = torch.mm(B, r)

            if e1 is not None:
                e1.synchronize()
            ABr = torch.mm(A, Br)

            if e3 is not None:
                e3.synchronize()

            Cr = torch.mm(C, r)

        ret = F.mse_loss(ABr, Cr).item()
        if ret > 1:
            print(f"freivalds_algorithm: {ret=}")
            print(f"Given C {C.shape} = {C}")
            print(f"Given A {A.shape} = {A}")
            print(f"Given B {B.shape} = {B}")
            print(f"cpu C = {torch.mm(A, B)}")
            print("Close: ", torch.allclose(C, torch.mm(A, B)))
            exit(-1)
        return ret
    # if not torch.allclose(ABr, Cr):
    #    ret = F.mse_loss(ABr, Cr).item()
    # return ret

def copy_to_cpu(x_device: torch.Tensor, stream_copy):
    if x_device.is_cuda:
        x_host = torch.empty_like(x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
        e = torch.cuda.Event()
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
        if self.linear.bias is not None:
            self.bias_cpu = linear.bias.clone().to("cpu")
            
        self.ctx = []
        self.cpu_stream = st_cpu
        self.gpu_stream = st_gpu 
        self.verify_event = torch.cuda.Event(blocking=True)
        self.compute_event = torch.cuda.Event(blocking=True)

    def forward(self, input):
        with torch.cuda.stream(self.gpu_stream):
            out = F.linear(input, self.linear.weight, bias=None)
            self.compute_event.record(self.gpu_stream)
            return out

    def add_bias(self, input):
        if self.linear.bias is not None:
            return input + self.linear.bias
        else:
            return input

    def add_bias_cpu(self, input):
        if self.linear.bias is not None:
            with torch.cuda.stream(self.cpu_stream):
                return input + self.bias_cpu
        else:
            return input

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
    test_verify(32, 64, 128)