from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer 
import torch

def get_c4_datasets(model_path, split="train"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    c4_streamed = load_dataset("/home/ubuntu/zg/data/c4", split=split)
    column_names = c4_streamed.column_names

    tokenized_datasets = c4_streamed.map(tokenize_function, remove_columns=column_names, num_proc=32, load_from_cache_file=True, batched=True)
    ds = C4Dataset(tokenized_datasets, tokenizer)
    return DataLoader(ds, shuffle=True, batch_size=1)

class C4Dataset(Dataset):
    def __init__(self, tokenized_datasets, tokenizer):
        self.tokenized_datasets = tokenized_datasets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenized_datasets)

    def __getitem__(self, idx):
        item = self.tokenized_datasets[idx]
        input_ids = torch.tensor(item['input_ids']).squeeze()
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift input_ids by one to the right for labels
        labels[-1] = self.tokenizer.eos_token_id  # Set the last label to the EOS token
        return input_ids, labels

def test_dataset():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataloader = get_c4_datasets(model_name)
    print(dataloader)
    for inp, lab in dataloader:
        print(inp)
        print(lab)
        exit(-1)

if __name__ == "__main__":
    test_dataset()