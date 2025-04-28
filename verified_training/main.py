import sys
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import torch.nn as nn

from torch.nn import Linear 
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim import AdamW
from verify_linear import VerifiedLinear, LinearWithVerification

from transformers import AutoTokenizer 
from transformers.models.llama import *
import torch.profiler
from torchmetrics.text import Perplexity
from utils.dataset_utils import get_c4_datasets

torch.set_num_threads(16)

def eval_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ppl_metric = Perplexity(ignore_index=tokenizer.pad_token_id).to('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_path)
    dataloader = get_c4_datasets(model_path, "test")

    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.cuda() 
            targets = targets.cuda() 

            outputs = model(input_ids)
            logits = outputs.logits  # [B, T, V]
            ppl_metric.update(logits, targets)

    final_ppl = ppl_metric.compute()
    print(f"Perplexity on C4 validation set: {final_ppl.item():.2f}")

def train(model, dataloader, num_epochs, batch_size, use_deepspeed=True, use_half = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==== using device {device}")
    model.to(device)
    #torch.autograd.set_detect_anomaly(True)
    if use_deepspeed:
        if use_half:
            config = "ds_config_fp16.json"
        else:
            config = "ds_config.json"
        
        if str(device) == "cpu":
            config = "ds_config_cpu.json"
            
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config
        )
    else:
        model_engine = model
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()               # Clear previous gradients

                print(">>> Run forward")
                st = time.perf_counter()
                
                outputs = model_engine(inputs)             # Forward pass
                fd_time = time.perf_counter() - st
                print(f"Forward time is {fd_time}")

                if isinstance(outputs, CausalLMOutputWithPast):
                    outputs = outputs.logits

                # print("####### checking nan")
                # print(outputs.isnan().any())
                # print(outputs)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

                print(">>> Run backward")
                st = time.perf_counter()
                if use_deepspeed:
                    model_engine.backward(loss)                     # Backward pass
                else:
                    loss.backward()
                fd_time = time.perf_counter() - st
                print(f"Backward Time is {fd_time}")

                print(">>> Run step")
                st = time.perf_counter()
                if use_deepspeed:
                    model_engine.step()                    # Update model parameters
                else:
                    optimizer.step()

                #prof.step()
                fd_time = time.perf_counter() - st
                print(f"Update Time is {fd_time}")
                break

        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


from verified_training.train import *
from verified_training.llm_model import create_llm_model
from verified_training.utils.log_utils import g_logger, logging
import os

def train_one_case(verify, use_ds=True, half=False, cpu_only=True, log="gpu.json"):
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        t = Train(model_name, get_c4_datasets(model_name), verify,
                   10, 1, use_ds, half, cpu_only, log)
        t.do()
    except Exception as e:
        g_logger.error(f"Run {verify=} {use_ds=} {half=} {cpu_only=} {log=} failed with {e}")
        return 


def main(use_ds, use_half):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) != "cpu":
        torch.cuda.reset_peak_memory_stats()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    replace_linear(model)
    dataloader = get_c4_datasets(model_name)
    train(model, dataloader, 3, 1, use_ds, use_half)
    LinearWithVerification.perf_log.process()
    LinearWithVerification.perf_log.to_csv("perf-improve.csv")

    if str(device) != "cpu":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak memory allocated: {peak_memory:.2f} GB")

if __name__ == "__main__":
    def _set(v, a, b):
        if v:
            return a
        else:
            return b

    import sys
    use_cpu = int(sys.argv[1])
    if use_cpu == 1:
        cpu_all = [True]
    else:
        cpu_all = [False]

    for cpu in cpu_all:
        c = _set(cpu, "cpu", "gpu")
        for verify in [True]:
            v = _set(verify, "v", "o")
            for use_ds in [True]:
                d = _set(use_ds, "ds", "nods")
                for half in [False]:
                    h = _set(half, "half", "fp32")
                    log_file = f"perfs/{v}-{d}-{h}-{c}.json"
                    print(log_file)
                    train_one_case(verify, use_ds, half, cpu, log_file)


    #main(False)
    #eval_model(model_name)
