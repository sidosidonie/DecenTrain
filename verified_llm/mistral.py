

#from huggingface_hub import snapshot_download
#from pathlib import Path
#
#mistral_models_path = Path.home().joinpath('mistral_models', '7B-v0.3')
#mistral_models_path.mkdir(parents=True, exist_ok=True)
#
#snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

# pip install "transformers>=4.42" accelerate torch --upgrade
# optional for 4-bit: pip install bitsandbytes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig, AutoConfig
from transformers.models.mistral3.modeling_mistral3 import *

model_id = "mistralai/Ministral-8B-Instruct-2410"
model_id = "mistralai/Mistral-7B-v0.3"  # or any other Mistral model

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained("./mistral-config.json")  # Load config if needed, or use default
print(config)
print(config._attn_implementation)

model = AutoModelForCausalLM.from_config(config)  # Use config to initialize the model


#model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    torch_dtype=torch.bfloat16,       # or torch.float16 if needed
#    device_map="auto",                # spreads across available GPUs/CPU
#    # load_in_4bit=True,              # uncomment if using bitsandbytes
#)
#
prompt = "You are a helpful assistant.\nUser: Give me 3 bullet points on SIMD.\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
out = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
#print(tokenizer.decode(out[0], skip_special_tokens=True))
