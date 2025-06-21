from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics.text import Perplexity
from verified_training.utils.dataset_utils import get_c4_datasets
import torch
from verified_training.utils.log_utils import g_logger

def eval_metrics(model, tokenizer, dataloader):
    ppl_metric = Perplexity(ignore_index=tokenizer.pad_token_id).to('cuda')

    with torch.no_grad():
        for input_data, targets in dataloader:
            input_ids = input_data["input_ids"].cuda()
            mask = input_data["attention_mask"].cuda()
            targets = targets.cuda() 

            outputs = model(input_ids, mask)
            logits = outputs.logits  # [B, T, V]
            ppl_metric.update(logits, targets)

    final_ppl = ppl_metric.compute()
    g_logger.info(f"Perplexity on C4 validation set: {final_ppl.item():.2f}")
