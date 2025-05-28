import sys
import os
from transformers.models.llama import LlamaAttention 

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import torch
from torch.autograd import Function
from torch.nn import Linear, Module, Parameter
from torch.nn import functional as F, init
from verified_training.verification import time_profile, freivalds_algorithm
import time
import pandas as pd
import deepspeed
from torch.utils.data import DataLoader, TensorDataset

class PerfLog:
    def __init__(self, use_half):
        self.log_dict = {
            "stage" : [],
            "shape": [],
            "compute_gpu" : [],
            "A_size_byte" : [],
            "B_size_byte" : [],
            "C_size_byte" : [],
            "A_to_cpu" : [],
            "B_to_cpu" : [],
            "C_to_cpu" : [],
            "verify_cpu" : [],
            "mse_loss": [],
            "cpu/gpu": [],
            "A_bandwidth" : [],
            "B_bandwidth" : [],
            "C_bandwidth" : [],
            "total": []
        }
        self.use_half = use_half

    @staticmethod
    def get_shape_key(input, weight):
        return f"{input.shape}-{weight.shape}"

    def record_perf(self, stage, input, weight, output, t, loss):
        self.log_dict["stage"].append(stage)
        self.log_dict["shape"].append(PerfLog.get_shape_key(input, weight))
        self.log_dict["compute_gpu"].append(t["compute_gpu"])
        self.log_dict["A_size_byte"].append(input.numel() * input.element_size()) 
        self.log_dict["B_size_byte"].append(weight.numel() * weight.element_size()) 
        self.log_dict["C_size_byte"].append(output.numel() * output.element_size()) 
        self.log_dict["A_to_cpu"].append(t["A_to_cpu"])
        self.log_dict["B_to_cpu"].append(t["B_to_cpu"])
        self.log_dict["C_to_cpu"].append(t["C_to_cpu"])
        self.log_dict["verify_cpu"].append(t["verify_cpu"])
        self.log_dict["mse_loss"].append(loss)
        self.log_dict["total"].append(t["compute_gpu"] + t["A_to_cpu"] + t["B_to_cpu"] + t["C_to_cpu"] + t["verify_cpu"])


    def process(self):
        self.log_dict["A_perc"] = []
        self.log_dict["B_perc"] = []
        self.log_dict["C_perc"] = []
        self.log_dict["compute_perc"] = []
        self.log_dict["verify_perc"] = []
        for i in range(len(self.log_dict["shape"])):
            self.log_dict["cpu/gpu"].append(self.log_dict["verify_cpu"][i] / self.log_dict["compute_gpu"][i])
            self.log_dict["A_bandwidth"].append(self.log_dict["A_size_byte"][i] / self.log_dict["A_to_cpu"][i] / 1e9)
            self.log_dict["B_bandwidth"].append(self.log_dict["B_size_byte"][i] / self.log_dict["B_to_cpu"][i] / 1e9)
            self.log_dict["C_bandwidth"].append(self.log_dict["C_size_byte"][i] / self.log_dict["C_to_cpu"][i] / 1e9)
            self.log_dict["A_perc"].append(self.log_dict["A_to_cpu"][i] / self.log_dict["total"][i] * 100)   
            self.log_dict["B_perc"].append(self.log_dict["B_to_cpu"][i] / self.log_dict["total"][i] * 100)   
            self.log_dict["C_perc"].append(self.log_dict["C_to_cpu"][i] / self.log_dict["total"][i] * 100)   
            self.log_dict["compute_perc"].append(self.log_dict["compute_gpu"][i] / self.log_dict["total"][i]*100) 
            self.log_dict["verify_perc"].append(self.log_dict["verify_cpu"][i] / self.log_dict["total"][i]*100) 
            
    def to_csv(self, fname):
        df = pd.DataFrame(self.log_dict)
        print(df)
        df.to_csv(fname)

def top_k_sparsify(grad, kp : float):
    flat_grad = grad.view(-1)
    k = int(flat_grad.numel() * kp)
    _, indices = torch.topk(flat_grad.abs(), k)
    
    sparse_grad = torch.zeros_like(flat_grad)
    sparse_grad[indices] = flat_grad[indices]
    
    return sparse_grad.view_as(grad)

def random_k_sparsify(grad, k):
    flat_grad = grad.view(-1)
    k = min(k, flat_grad.numel())
    indices = torch.randperm(flat_grad.numel())[:k]
    
    sparse_grad = torch.zeros_like(flat_grad)
    sparse_grad[indices] = flat_grad[indices]
    
    return sparse_grad.view_as(grad)


class LinearWithVerification(Function):
    
    perf_log = PerfLog(True)

    @staticmethod
    def copy_sparse_and_perf(x_device : torch.Tensor):
        if str(x_device.device) == "cpu":
            return x_device, 0

        st = time.perf_counter()
        sp = top_k_sparsify(x_device, 0.1).to_sparse()
        x_host = torch.empty_like(sp, device="cpu", pin_memory=True, dtype=x_device.dtype)
        xx = sp.clone()
        xx.copy_(x_host)
        ed = time.perf_counter()
        return x_host, ed - st

    @staticmethod
    def copy_and_perf(x_device : torch.Tensor):
        if str(x_device.device) == "cpu":
            return x_device, 0

        st = time.perf_counter()
        x_host = torch.empty_like(x_device, device="cpu", pin_memory=True, dtype=x_device.dtype)
        # x_half = x_device.to(dtype=torch.half)
        xx = x_device.clone()
        xx.copy_(x_host)
        ed = time.perf_counter()
        return x_host, ed - st

    @staticmethod
    def copy_and_verify_with_perf(input, weight, output, t, desc):
        input_cpu, input_time = LinearWithVerification.copy_and_perf(input)
        weight_cpu, weight_time = LinearWithVerification.copy_and_perf(weight)
        output_cpu, output_time = LinearWithVerification.copy_and_perf(output)
        loss, verify_time = LinearWithVerification.verify(input_cpu, weight_cpu, output_cpu)

        # print(f"input dtype {input.dtype}")
        # print(f"weight dtype {weight.dtype}")
        # print(f"output dtype {output.dtype}")

        t["A_to_cpu"] = input_time
        t["B_to_cpu"] = weight_time
        t["C_to_cpu"] = output_time
        t["verify_cpu"] = verify_time

        LinearWithVerification.perf_log.record_perf(desc, input, weight, output, t, loss)
        return input_cpu, weight_cpu, output_cpu

    @staticmethod
    def linear_and_perf(input, weight, t):
        st = time.perf_counter()
        output = F.linear(input, weight)
        ed = time.perf_counter()
        t["compute_gpu"] = ed - st
        return output

    def matmul_and_perf(input, weight, t):
        st = time.perf_counter()
        output = torch.matmul(input, weight)
        ed = time.perf_counter()
        t["compute_gpu"] = ed - st
        return output

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        t = {}

        output = LinearWithVerification.linear_and_perf(input, weight, t)
        LinearWithVerification.copy_and_perf(output)
        
        #LinearWithVerification.copy_and_verify_with_perf(input, weight.t(), output, t, "forward")
        LinearWithVerification.verify(input, weight.t(), output)

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Save tensors for backward computation
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias= ctx.saved_tensors

        # Compute gradients
        t_grad = {}
        grad_input = LinearWithVerification.matmul_and_perf(grad_output, weight, t_grad)
        LinearWithVerification.copy_and_perf(grad_input)
        #LinearWithVerification.copy_and_verify_with_perf(grad_output, weight, grad_input, t_grad, "backward_grad")
        LinearWithVerification.verify(grad_output, weight, grad_input)

        # Compute weights
        t_weight = {}
        grad_out_t = grad_output.transpose(-2, -1)
        grad_weight = LinearWithVerification.matmul_and_perf(grad_out_t, input, t_weight)
        LinearWithVerification.copy_and_perf(grad_weight)
        LinearWithVerification.verify(grad_out_t, input, grad_weight)
        #LinearWithVerification.copy_and_verify_with_perf(grad_out_t, input, grad_weight, t_weight, "backward_weight")

        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias

    @staticmethod
    def verify(A, B, C):
        st = time.perf_counter()
        A = torch.empty_like(A, device="cpu")
        B = torch.empty_like(B, device="cpu")
        C = torch.empty_like(C, device="cpu")
        loss = freivalds_algorithm(A, B, C)
        ed = time.perf_counter()
        return loss, ed-st
        #if loss <= 1e-5:
        #    pass
        #    #print(f"Verify success! {A.min()=}-{A.max()} {B.min()=}-{B.max()} {C.min()=}-{C.max()}")
        #else:
        #    print(f"Verify Failed! MSE loss = {loss}: {A.min()=}-{A.max()} {B.min()=}-{B.max()} {C.min()=}-{C.max()}")

        #return loss
            
            #raise RuntimeError("Verify matmul failed!")


# Custom Linear Module using the LinearWithVerification Function
class VerifiedLinear(Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        super(VerifiedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = Parameter(torch.randn(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)


    def forward(self, input):
        return LinearWithVerification.apply(input, self.weight, self.bias)


class TestModule(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(TestModule, self).__init__()
        self.linear1 = VerifiedLinear(in_features, out_features)
        self.linear2 = VerifiedLinear(out_features, in_features)

    def forward(self, input):
        a = self.linear1(input)
        return self.linear2(a)

if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inc = 1024
    outc = 1024
    batch = 128

    model = TestModule(inc, outc).to(device)
    x = torch.randn(batch, inc)
    
    

    print(top_k_sparsify(x, 0.1).to_sparse())
    exit(-1)


    y = torch.randn(batch, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)

    #input_tensor_ref = input_tensor.clone() 
    #input_tensor_ref.requires_grad_(True)
    #input_tensor_ref.retain_grad()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_fp16.json"
    )

    criterion = torch.nn.MSELoss()

    for epoch in range(5):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(model_engine.local_rank, dtype=torch.float16)
            batch_y = batch_y.to(model_engine.local_rank, dtype=torch.float16)

            outputs = model_engine(batch_x)
            predict = outputs.sum(-1, keepdim=True)
            loss = criterion(predict, batch_y)

            model_engine.backward(loss)
            model_engine.step()

    print(f"Peak memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print("successful")
    #output.sum().backward()

    #ref_model = Linear(inc, outc, bias=False).to(device)
    #ref_model.weight = model.weight

    #output_ref = ref_model(input_tensor_ref)
    #output_ref.sum().backward()

    LinearWithVerification.perf_log.process()
    LinearWithVerification.perf_log.to_csv("verify_perf_fp32.csv")
