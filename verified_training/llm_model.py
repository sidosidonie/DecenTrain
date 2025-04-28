"""
  Build llm models with verified linear
"""
from torch.nn import Linear 
from transformers import AutoModelForCausalLM
from verified_training.verify_linear import VerifiedLinear 
from verified_training.utils.log_utils import g_logger

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

