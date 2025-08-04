import time
import math
from functools import wraps
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter
import inspect

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
