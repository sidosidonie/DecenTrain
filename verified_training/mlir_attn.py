from transformers.utils import import_utils
from transformers.models.llama import LlamaAttention, LlamaConfig
import torch
from torch_mlir import fx, compiler_utils
from transformers import pipelines

from torch_mlir.dialects import arith, func
from torch_mlir.dialects import torch as torchop 
from torch_mlir.ir import *
from torch_mlir.passmanager import PassManager
from torch_mlir.extras import types 

from verified_training.utils.log_utils import g_logger

def get_num_elem(t : RankedTensorType):
    """
    Get the number of elements in a tensor type.
    """
    num_elem = 1
    for dim in t.shape:
        if dim == -1:
            return -1
        num_elem *= dim
    return num_elem

def replace_constants_with_zero(module: Module):
    """
    In-place modifies the module by replacing all constants with zeros.
    """
    def callback(op):
        if isinstance(op, arith.ConstantOp):
            print(type(op))

            # Replace scalar constants with zero
       #     zero_attr = IntegerAttr.get(op.value.type, 0)
       #     op.value = zero_attr
       # elif isinstance(op, torchop.ConstantOp):
       #     # Replace torch constant tensors with zeros
       #     orig_attr = op.value
       #     if isinstance(orig_attr, DenseElementsAttr):
       #         zero_data = [0] * orig_attr.num_elements
       #         zero_attr = DenseElementsAttr.get(
       #             orig_attr.type, zero_data
       #         )
       #         op.value = zero_attr

    def replace_dense_resource_elements_attr(origin_attr, value):
        if DenseResourceElementsAttr.isinstance(origin_attr):
            g_logger.info(f"Replacing constant origin_attr with zero")
            attr = DenseResourceElementsAttr(cast_from_attr=origin_attr)
            g_logger.info(f"attr: {attr.type}")
            zero_attr = DenseElementsAttr.get_splat(attr.type, value)
            return zero_attr
        else:
            return None

    def replace_arith_constant(op):
        origin_attr = op.value
        val = FloatAttr.get_f32(1.0)
        new_attr = replace_dense_resource_elements_attr(origin_attr, val)
        if new_attr is None:
            return
        else:
            loc = op.operation.location
            with InsertionPoint(op):
                if isinstance(op, arith.ConstantOp):
                    new_op = arith.constant(type=new_attr.type, value=new_attr, loc=loc)
                elif isinstance(op, torchop.ValueTensorLiteralOp):
                    g_logger.info("Replacing torch constant with zero")
                    new_op = torchop.vtensor_literal(value=new_attr, loc=loc)
                else:
                    raise ValueError(f"Unsupported op type: {type(op)}")

                op.operation.result.replace_all_uses_with(new_op)
                op.operation.erase()

    with module.context:
        for f in module.body:
            if isinstance(f, func.FuncOp):
                for blk in f.body:
                    for op in blk:
                        if isinstance(op, arith.ConstantOp) or isinstance(op, torchop.ValueTensorLiteralOp):
                            replace_arith_constant(op)


def register_zero_constants_pass():
    def pass_fn(module):
        replace_constants_with_zero(module)
    return pass_fn

def original_attn(config, batch, seq_len):
    attn = LlamaAttention(config, 0)

    # Typical LLaMA config values
    batch_size = batch
    seq_length = seq_len
    hidden_size = config.hidden_size  # varies by model size
    num_heads = config.num_attention_heads  # e.g., for LLaMA-7B
    head_dim = hidden_size // num_heads  # e.g., 128 // 16 = 8

    # Fake hidden_states
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)

    # Fake attention mask (optional)
    # LLaMA typically uses causal mask internally, but some versions accept this
    attention_mask = torch.ones(batch_size, 1, 1, seq_length)

    # Fake position_ids (optional)
    cos = torch.randn(batch_size, seq_length, head_dim)
    sin = torch.randn(batch_size, seq_length, head_dim)
    position_ids = torch.stack([cos, sin], dim=0)


    print("hidden_states:", hidden_states.shape)
    print("attention_mask:", attention_mask.shape)
    print("position_ids:", position_ids.shape)

    # graph, guards = torch._dynamo.export(compiled, hidden_states, position_ids, attention_mask)

    #compiled_attention = torch.compile(compiled, backend=debug_backend)
    # compiled_attention(hidden_states, position_ids, attention_mask)

    #m = fx.export_and_import(attn, hidden_states, position_ids, None, func_name="test_net", output_type=compiler_utils.OutputType.TORCH)
    return None, attn, (hidden_states, position_ids, None)

def replace_const(m):
    with m.context:
        replace_constants_with_zero(m)
        #pm = PassManager.parse("builtin.module()")
        #pm.add_pass(register_zero_constants_pass())
        #pm.run(m.operation)

if __name__ == "__main__":
    mm = original_attn()
    with open("attn-origin.mlir", "w") as f:
        f.write(str(mm))

    replace_const(mm)
    with open("attn.mlir", "w") as f:
        f.write(str(mm))
