from llm_model import create_llm_model
from utils.dataset_utils import get_c4_datasets
from transformers import AutoTokenizer
from verified_training.eval import eval_metrics
import torch

def train(batch, seqlen):
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.default_stream()
    model = create_llm_model("meta-llama/Llama-3.2-1B-Instruct",
                             verify=True, cpu=cpu_stream, gpu=gpu_stream)
    data = get_c4_datasets("meta-llama/Llama-3.2-1B-Instruct", batch, seqlen, "train")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eval_metrics(model, tokenizer, data)


train(8, 1024)
