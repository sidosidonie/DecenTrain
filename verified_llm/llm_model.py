"""
  Build llm models with verified linear
"""
from transformers import AutoModelForCausalLM
from verified_llm.mlp_layer import LlamaMLPVerify, LlamaMLP
from verified_llm.attn_layer import LlamaAttentionVerify , LlamaAttention
from verified_llm.utils.log_utils import g_logger
from verified_llm.utils.profiler import Profiler 
import torch
import functools

def replace_param_attn(origin : LlamaAttention, stream_cpu, stream_gpu, noise):
    new_linear = LlamaAttentionVerify(origin, stream_cpu, stream_gpu, noise)
    return new_linear

def replace_attn(model, cpu, gpu, noise):
    for name, module in model.named_children():
        if isinstance(module, LlamaAttention):
            veri_mlp = replace_param_attn(module, cpu, gpu, noise)
            setattr(model, name, veri_mlp)
        else:
            replace_attn(module, cpu, gpu, noise)

def replace_param_mlp(origin : LlamaMLP, stream_cpu, stream_gpu, noise):
    new_linear = LlamaMLPVerify(origin, stream_cpu, stream_gpu, noise)
    return new_linear

def replace_mlp(model, cpu, gpu, noise):
    for name, module in model.named_children():
        if isinstance(module, LlamaMLP):
            veri_mlp = replace_param_mlp(module, cpu, gpu, noise)
            setattr(model, name, veri_mlp)
        else:
            replace_mlp(module, cpu, gpu, noise)


def dump_layer_outputs(model):
    layer_outputs = {}

    def hook_fn(name, module, input, output):
        layer_outputs[name] = output

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            hooks.append(module.register_forward_hook(functools.partial(hook_fn, name)))
    return layer_outputs, hooks

def create_llm_model(model_path, verify=False, cpu=None, gpu=None, noise=None):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config._attn_implementation = "eager"
    model.to("cuda")
    p = Profiler()
    dur = p.add_time_span("attn")
    if verify:
        replace_attn(model, cpu, gpu, noise)
        replace_mlp(model, cpu, gpu, noise)

    return model

if __name__ == "__main__":
    cpu = torch.cuda.Stream()
    gpu = torch.cuda.default_stream()
    mod = create_llm_model("meta-llama/Llama-3.2-1B-Instruct", True, cpu, gpu)
    print(mod)