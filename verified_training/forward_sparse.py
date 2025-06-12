from verified_training.utils.dataset_utils import get_c4_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import functional as F, init
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math
from llm_model import create_sparse_llm_model


def load_model(model_path):
    llm = AutoModelForCausalLM.from_pretrained(model_path)
    print(llm.config)
    return llm

def test_sparse_weight1():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## prepare model
    llm = load_model(model_name)
    prompt = "What is the capital of France?"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=2048)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=50, do_sample=True, return_dict_in_generate=True)
        print(outputs)
        print(outputs.sequences)
        decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(decoded)


def test_sparse_weight():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## prepare model
    llm = create_sparse_llm_model(model_name)

    dataloader = get_c4_datasets(model_name)
    print(dataloader)
    for inp, lab in dataloader:
        decode_input = tokenizer.batch_decode(inp["input_ids"], skip_special_tokens=True)
        print(decode_input)
        print(inp)
        print(lab)

        with torch.no_grad():
            #outputs = llm(**inp, labels=lab)
            outputs = llm.generate(**inp, max_new_tokens=50, do_sample=True)
            decoded = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
            print(decoded)

        break

def generate_answer(prompt, model_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=2048)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("anser is ", answer)
    return answer


test_sparse_weight()