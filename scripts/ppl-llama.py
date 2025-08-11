from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import math
from tqdm import tqdm
from verified_llm.llm_model import create_llm_model
from verified_llm.utils.log_utils import g_logger
import argparse

g_logger.setLevel("INFO")

parser = argparse.ArgumentParser()
parser.add_argument("--noise", type=float, default=None)
parser.add_argument("--limit_samples", type=int, default=100)
args = parser.parse_args()

# ----------- CONFIG ----------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # or any other HF-compatible model
data_path = "/home/ecs-user/data/c4"
split = "test"                           # Can also be "train"
limit_samples = args.limit_samples                     # For demo. Set to None for full eval
batch_size = 4  # Set your desired batch size
# -----------------------------------

g_logger.info("Starting perplexity calculation")
g_logger.info(f"Model: {model_name}")
g_logger.info(f"Data path: {data_path}")
g_logger.info(f"Split: {split}")
g_logger.info(f"Limit samples: {limit_samples}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(
#    model_name, torch_dtype=torch.float16, device_map="auto"
#)
cpu_stream = torch.cuda.Stream()
gpu_stream = torch.cuda.default_stream()
model = create_llm_model(model_name, verify=True, cpu=cpu_stream, gpu=gpu_stream, noise=args.noise)
model.eval()

# Load dataset from local JSONL file
dataset = load_dataset("json", data_files={split: f"{data_path}/{split}.jsonl"})[split]

if limit_samples:
    dataset = dataset.select(range(limit_samples))


for context_len in [512]:
    total_loss = 0.0
    total_tokens = 0
    max_length = context_len
    stride = context_len

    input_chunks = []
    label_chunks = []

    sample_i = 0
    for example in tqdm(dataset):
        g_logger.info(f"Processing example {sample_i}")
        sample_i += 1
        text = example["text"]
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids[0]
        if len(input_ids) < max_length:
            continue  # skip short samples

        for i in range(0, len(input_ids) - max_length + 1, stride):
            input_chunk = input_ids[i : i + max_length]
            labels = input_chunk.clone()
            input_chunks.append(input_chunk)
            label_chunks.append(labels)

            # When enough samples are collected, process as a batch
            if len(input_chunks) == batch_size:
                inputs = {
                    "input_ids": torch.stack(input_chunks).to(model.device),
                    "labels": torch.stack(label_chunks).to(model.device)
                }
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss
                    total_loss += loss.item() * sum(chunk.size(0) for chunk in input_chunks)
                    g_logger.info(f"Batch loss: {loss.item()}, total loss: {total_loss}")
                    total_tokens += sum(chunk.size(0) for chunk in input_chunks)
                input_chunks = []
                label_chunks = []

    # Process any remaining samples
    #if input_chunks:
    #    inputs = {
    #        "input_ids": torch.stack(input_chunks).to(model.device),
    #        "labels": torch.stack(label_chunks).to(model.device)
    #    }
    #    with torch.no_grad():
    #        outputs = model(**inputs)
    #        loss = outputs.loss
    #        total_loss += loss.item() * sum(chunk.size(0) for chunk in input_chunks)
    #        total_tokens += sum(chunk.size(0) for chunk in input_chunks)
    #    input_chunks = []
    #    label_chunks = []

    # Final perplexity
    if total_tokens > 0:
        avg_neg_log_likelihood = total_loss / total_tokens
        g_logger.info(f"Avg neg log likelihood: {avg_neg_log_likelihood}")
        perplexity = math.exp(avg_neg_log_likelihood)
        g_logger.info(f"Perplexity for n_ctx: {max_length} on local {split}.jsonl: {perplexity:.2f}")
    else:
        print("No valid samples processed.")
