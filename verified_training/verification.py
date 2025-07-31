import time
import math
from functools import wraps
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter
import inspect


def time_profile(log_file=None, log_dict=None):

    def _time_profile(func):
        """
        Decorator that measures the execution time of a function.

        Usage:
            @time_profile
            def my_function(...):
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"[{func.__name__}] Execution time: {duration:.6f} seconds")

            if log_file is not None:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                log_file.write(f"[{func.__name__}] ")
                for k, v in bound_args.arguments.items():
                    if isinstance(v, torch.Tensor):
                        log_file.write(f"{k}={v.shape} ")
                    else:
                        log_file.write(f"{k}={v} ")

                log_file.write(f"--> {duration:.6f} seconds\n")

            return result
        return wrapper

    return _time_profile


log_file = open("log.perf", "a+")


@time_profile()
def freivalds_algorithm_sparse(A, B, C, k=10):
    n = C.shape[-1]
    r = random_sparse_matrix(n, k, dtype=A.dtype)

    Br = B @ r
    ABr = A @ Br
    Cr = C @ r

    if not torch.allclose(ABr, Cr):
        loss = F.mse_loss(ABr, Cr).item()
        return loss

    return 0


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
    print(f"{A.shape=}, {A.device}")
    print(f"{B.shape=}, {B.device}")
    print(f"{C.shape=}, {C.device}")

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

# @time_profile()


def freivalds_algorithm(A, B, C, stream, e1=None, e2=None, e3=None, k=10):
    # print(f"{A.shape=}, {A.device}")
    # print(f"{B.shape=}, {B.device}")
    # print(f"{C.shape=}, {C.device}")
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
        return ret
        if ret > 1:
            print(f"freivalds_algorithm: {ret=}")
            print(f"ABr = {ABr}")
            print(f"Cr {Cr.shape=} = {Cr}")
            print(f"r {r.shape=} = {r}")
            print(f"A {A.shape=} = {A}")
            print(f"B {B.shape=} = {B}")
            print(f"gpu C {C.shape} = {C}")
            print(f"cpu C = {torch.matmul(A, B)}")
            exit(-1)
        return ret
    # if not torch.allclose(ABr, Cr):
    #    ret = F.mse_loss(ABr, Cr).item()
    # return ret


def random_sparse_matrix(rows, cols, nnz_perc=0.6, dtype=torch.float32):
    """
    Generate a random sparse COO matrix with given shape and number of non-zero elements.
    """
    # Randomly choose indices (without duplicates)
    row_num = int(nnz_perc * rows)
    indices = torch.empty(2, row_num*cols)

    for c in range(cols):
        r = torch.randperm(rows)[0:row_num]
        for rr in range(row_num):
            indices[0, c*row_num+rr] = r[rr]
            indices[1, c*row_num+rr] = c

    # Generate random values for the non-zero entries
    values = torch.randn(indices.shape[1], dtype=dtype)

    # Create sparse tensor
    sparse = torch.sparse_coo_tensor(indices, values, (rows, cols))
    return sparse.coalesce()  # Coalesce to remove any duplicate entries and sum them


@time_profile(log_file)
def freivalds_algorithm_origin(A, B, C, k=10):
    """
    Probabilistically verify whether A @ B == C using Freivalds' algorithm.

    Args:
        A (torch.Tensor): Matrix of size (n x n)
        B (torch.Tensor): Matrix of size (n x n)
        C (torch.Tensor): Matrix of size (n x n)
        k (int): Number of iterations 

    Returns:
        bool: True if the matrices likely satisfy A @ B == C, False otherwise.
    """
    n = C.shape[-1]
    for _ in range(k):
        r = torch.randn((n, 1), dtype=A.dtype, device=A.device)
        # Compute A(B r) and C r
        Br = B @ r
        ABr = A @ Br
        Cr = C @ r

        # If mismatch is found, reject
        if not torch.allclose(ABr, Cr):
            return F.mse_loss(ABr, Cr).item()

    return 0


@time_profile()
def transfer_to_gpu(A):
    assert str(A.device) == "cpu"
    AA = A.to("cuda")
    torch.cuda.synchronize()
    return AA


@time_profile()
def transfer_to_cpu(A):
    assert A.is_cuda
    AA = A.to("cpu")
    torch.cuda.synchronize()
    return AA


@time_profile()
def matmul_on_gpu(A, B):
    assert A.is_cuda
    assert B.is_cuda
    CC = A @ B
    return CC


class VerifiedLinear(nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

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
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, **factory_kwargs))
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
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def test_sparse():
    sparse = random_sparse_matrix(10, 5)
    print(sparse)
    dense = torch.rand(10)
    res = dense @ sparse
    print(res)


def main(dt=torch.float32):
    print(f"======= test type {dt}")
    n = 8192
    m = 8192
    k = 2048
    # A = torch.randn(n, n)
    # B = torch.randn(n, n)
    # mxk kxn -> mxn
    # kxn @ nx1 -> kx1
    # mxk @ kx1 -> mx1
    # mxn @ nx1 -> mx1
    A = torch.randn((m, k), dtype=dt)
    B = torch.randn((n, k), dtype=dt)

    print("start do matmul")
    print(A.dtype)
    AA = transfer_to_gpu(A)
    BB = transfer_to_gpu(B)
    print(AA.dtype)
    print(BB.dtype)
    CC = matmul_on_gpu(AA, BB.t())
    C = transfer_to_cpu(CC)

    print("Correct product check:", freivalds_algorithm(
        A, B.t(), C))          # Expected: True
    # print("Correct product check:", freivalds_algorithm_origin(A, B.t(), C))          # Expected: True
    # print("Correct product check:", freivalds_algorithm_sparse(A, B.t(), C))          # Expected: True
    # print("Incorrect product check:", freivalds_algorithm(A, B, C_wrong))  # Expected: False (most of the time)
    log_file.close()


if __name__ == "__main__":
    for _ in range(1):
        main()
        main(torch.half)
    # main(torch.half)
    # test_sparse()
