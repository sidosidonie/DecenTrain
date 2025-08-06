from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_model import create_llm_model, dump_layer_outputs, LlamaConfig
from utils.dataset_utils import get_c4_datasets
from transformers import AutoTokenizer
from torchmetrics.text import Perplexity
from verified_llm.utils.dataset_utils import get_c4_datasets
from verified_llm.utils.log_utils import g_logger
import torch
from fire import Fire
import os
import numpy as np
import logging
from pprint import pprint

g_logger.setLevel(logging.INFO)

def eval_metrics(model, tokenizer, dataloader, itern=1):
    ppl_metric = Perplexity(ignore_index=tokenizer.pad_token_id).to('cuda')

    with torch.no_grad():
        i = 0
        for input_data, targets in dataloader:
            input_ids = input_data["input_ids"].cuda()
            mask = input_data["attention_mask"].cuda()
            targets = targets.cuda() 
            outputs = model(input_ids, mask)
            logits = outputs.logits  # [B, T, V]
            ppl_metric.update(logits, targets)

            i += 1
            if i == itern:
                break

    final_ppl = ppl_metric.compute()
    g_logger.info(f"Perplexity on C4 validation set: {final_ppl.item():.2f}")

def forward(input_len, verify, itern = 10):
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.default_stream()
    model = create_llm_model(model_path, verify=verify, cpu=cpu_stream, gpu=gpu_stream)
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(1, input_len),
        dtype=torch.long,
        device="cuda"
    )
    attention_mask = torch.ones_like(input_ids, device="cuda")
    with torch.no_grad():
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        total_time = 0
        for i in range(itern):
            st.record()
            outputs = model(input_ids, attention_mask)
            torch.cuda.synchronize()
            ed.record()
            elapsed_time = st.elapsed_time(ed)
            print(f"Iter {i}: {elapsed_time}")
            if i > 0:
                total_time += elapsed_time

        print(f"Forward pass time: {total_time/(itern-1)} ms")
        return outputs, elapsed_time

def generate(prompt, verify, dump = False):
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.default_stream()

    model_verify = create_llm_model(model_path, verify=verify, cpu=cpu_stream, gpu=gpu_stream)
    if dump:
        layer_outputs, hooks = dump_layer_outputs(model_verify)

    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.cuda()
    with torch.no_grad():
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)

        st.record()
        out_toks = model_verify.generate(input_ids, max_new_tokens=10, do_sample=False, temperature=0.1)
        ed.record()
        total_time = st.elapsed_time(ed)

        generated_text = tokenizer.batch_decode(out_toks, skip_special_tokens=True)
        return generated_text, total_time

def eval(batch = 8, seqlen = 1024):
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.default_stream()
    model = create_llm_model("meta-llama/Llama-3.2-1B-Instruct",
                             verify=False, cpu=cpu_stream, gpu=gpu_stream)
    data = get_c4_datasets("meta-llama/Llama-3.2-1B-Instruct", batch, seqlen, "train")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eval_metrics(model, tokenizer, data)

if __name__ == "__main__":
    prompts = [
    "Once upon a time, there was",
    "In a future world ruled by AI,",
    "The recipe for the perfect cake is" 
    ]
    len = 1024*3
    gen_text_ori, time_ori = generate(prompts, True, False)
    print(f"Origin generation time: {time_ori} ms")
    for gen_text in gen_text_ori:
        print(f">>> Origin generated text: {gen_text}")

    #gen_text_ver, time_ver = generate(prompt, True, False)
    #print(f">>> Verify generated text: {gen_text_ver}, use time: {time_ver} ms")

    #gen_text_ver, time_ver = forward(len, True)
    #print(f"Verify: {time_ver}ms")
    #print(f"Origin: {time_ori}ms")