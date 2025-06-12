"""
  Build llm models with verified linear
"""
from torch.nn import Linear 
from transformers import AutoModelForCausalLM
from verified_training.verify_linear import VerifiedLinear 
from verified_training.utils.log_utils import g_logger
import torch
from torch.nn import functional as F, init
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math

def print_tensor_distribution(tensor, tol=1e-6):
    total = tensor.numel()
    close_to_zero = (tensor.abs() < tol).sum().item()
    print(f"Total elements: {total}")
    print(f"Elements close to zero (|x| < {tol}): {close_to_zero} ({close_to_zero/total:.2%})")
    print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}, Std: {tensor.std().item()}")


class SparseLinear(torch.nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

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
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.id = None

    def set_id(self, _id):
        self.id = _id

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
        out = F.linear(input, self.weight, self.bias)
        print_tensor_distribution(out, 1e-2)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def replace_sparse_linear_and_add_name(mod : torch.nn.Module):
    # set name
    for name, module in mod.named_modules():
        if isinstance(module, torch.nn.Linear):
            print("Module name is ", name)
            setattr(module, "layer_name", name)

    for name, module in mod.named_children():
        if isinstance(module, torch.nn.Linear):
            print("replace ", name)
            sp_linear = SparseLinear(module.in_features, module.out_features, module.bias)
            sp_linear.weight = module.weight
            sp_linear.bias = module.bias
            sp_linear.id = module.layer_name
            setattr(mod, name, sp_linear)
        else:
            replace_sparse_linear_and_add_name(module)

    print("====== after replace ment")
    for name, module in mod.named_modules():
        if isinstance(module, torch.nn.Linear):
            print("Module name is ", name)


def replace_param(origin : Linear):
    new_linear = VerifiedLinear(origin.in_features, origin.out_features, origin.bias)
    new_linear.weight = origin.weight
    if origin.bias:
        new_linear.bias = origin.bias
    return new_linear

def replace_linear(model):
    for name, module in model.named_children():
        if isinstance(module, Linear):
            veri_linear = replace_param(module)
            setattr(model, name, veri_linear)
        else:
            replace_linear(module)

def create_llm_model(model_path, verify=False):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if verify:
        g_logger.debug("Creating verified LLM")
        replace_linear(model)

    return model


def create_sparse_llm_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    g_logger.debug("Creating verified LLM")
    replace_sparse_linear_and_add_name(model)
    return model


if __name__ == "__main__":
    mod = create_sparse_llm_model("meta-llama/Llama-3.2-1B-Instruct")
    print(mod)