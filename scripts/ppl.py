from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
if torch.cuda.is_available():
    model.cuda()

# Load evaluation text
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.4f}")
