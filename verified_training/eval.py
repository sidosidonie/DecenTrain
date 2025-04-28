from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics.text import Perplexity
from verified_training.utils.dataset_utils import get_c4_datasets
import torch
from verified_training.utils.log_utils import g_logger

def eval_metrics(model_path):
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
    g_logger.info(f"Perplexity on C4 validation set: {final_ppl.item():.2f}")

def generate_sentence():
    pass