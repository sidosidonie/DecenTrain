from llm_model import create_llm_model
from utils.dataset_utils import get_c4_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# Generate tokens
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)

# Decode generated tokens to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("---- generated text:")
print(generated_text)

