import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import deepspeed

from torch.nn import Linear, Module, Parameter
from transformers import AutoModelForCausalLM 
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from verify_linear import VerifiedLinear, LinearWithVerification

from transformers import AutoTokenizer

torch.cuda.reset_peak_memory_stats()
custom_linear = Linear

def wiki_dataset(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-v1")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True, num_proc=4)
    return tokenized_datasets


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.weight * x / (norm + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb


def apply_rotary_pos_emb(x, rope):
    x1, x2 = x[..., ::2], x[..., 1::2]
    rope_sin, rope_cos = rope[..., ::2], rope[..., 1::2]
    x_rotated = torch.cat([x1 * rope_cos - x2 * rope_sin,
                           x1 * rope_sin + x2 * rope_cos], dim=-1)
    return x_rotated

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, dropout=0.0):
        super(GroupedQueryAttention, self).__init__()
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.dropout = nn.Dropout(dropout)

        self.W_q = custom_linear(embed_dim, embed_dim, bias=False)
        self.W_k = custom_linear(embed_dim, embed_dim // num_query_heads * num_kv_heads, bias=False)
        self.W_v = custom_linear(embed_dim, embed_dim // num_query_heads * num_kv_heads, bias=False)
        self.W_o = custom_linear(embed_dim, embed_dim, bias=False)

    
    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.W_q(x).view(B, T, self.num_query_heads, self.head_dim)
        k = self.W_k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.W_v(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, num_kv_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, num_kv_heads, T, head_dim)

        # Scale dot-product attention
        k = GroupedQueryAttention.repeat_kv(k, 4)
        v = GroupedQueryAttention.repeat_kv(v, 4)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.W_o(attn_output)
        return output


class LlamaMLP(nn.Module):
    def __init__(self, input_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = custom_linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = custom_linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = custom_linear(intermediate_dim, input_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, kv_heads, ff_dim):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, n_heads, kv_heads)
        self.ff = LlamaMLP(dim, ff_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class CustomLLaMAModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_query_heads, num_kv_heads, intermediate_size, max_position_embeddings):
        super(CustomLLaMAModel, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_query_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.output = custom_linear(hidden_size, vocab_size, bias=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        # todo: just leave rotery for now
        logits = self.output(x)
        return logits


# Load the configuration file
def custom_model(model_path):
    config_path = model_path + "config.json" 
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        print(config)

    # Extract relevant parameters
    vocab_size = config.get('vocab_size', 32000)
    hidden_size = config.get('hidden_size', 4096)
    num_layers = config.get('num_hidden_layers', 32)
    num_query_heads = config.get('num_attention_heads', 32)
    num_kv_heads = config.get('num_key_value_heads', 8)  # Example value; adjust as per actual config
    intermediate_size = config.get('intermediate_size', 11008)
    max_position_embeddings = config.get('max_position_embeddings', 2048)
    # Initialize the model
    model = CustomLLaMAModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings
    )

    return model

#scaler = GradScaler("cuda")


def train(model, dataloader, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    torch.autograd.set_detect_anomaly(True)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            #inputs = torch.randint(low=0, high=10000, size=(batch_size, 512))
            #targets = inputs.clone() 
            inputs = torch.stack(batch["input_ids"], dim = 0)
            inputs = inputs.t()
            targets = torch.stack(batch["labels"], dim = 0) 
            targets = targets.t()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()               # Clear previous gradients

            st = time.perf_counter()
            outputs = model_engine(inputs)             # Forward pass
            fd_time = time.perf_counter() - st
            print(f"Forward time is {fd_time}")

            if isinstance(outputs, CausalLMOutputWithPast):
                outputs = outputs.logits

            #loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            st = time.perf_counter()
            model_engine.backward(loss)                     # Backward pass
            fd_time = time.perf_counter() - st
            print(f"Backward Time is {fd_time}")

            st = time.perf_counter()
            model_engine.step()                    # Update model parameters
            fd_time = time.perf_counter() - st
            print(f"Update Time is {fd_time}")
            break

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


def get_datasets(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    c4_streamed = load_dataset("../data/c4", split="train")
    column_names = c4_streamed.column_names

    tokenized_datasets = c4_streamed.map(tokenize_function, remove_columns=column_names, num_proc=16, load_from_cache_file=True)

    def shift_tokens(examples):
        input_ids = examples["input_ids"] #.to(dtype=torch.float16)
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]  # Shift input_ids by one to the right for labels
        labels[-1] = tokenizer.eos_token_id  # Set the last label to the EOS token
        examples["labels"] = labels
        return examples

    lm_datasets = tokenized_datasets.map(shift_tokens, num_proc=16, load_from_cache_file=True)
    return DataLoader(lm_datasets, shuffle=True, batch_size=4)

def ref_model(loader, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    train(model, loader, 1, 4)

def main():
    model_path = '/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/'

    loader = get_datasets(model_path)

    print("======= Verified Llama =======")
    # model = custom_model(model_path)
    # train(model, loader, 1, 4)
    # LinearWithVerification.perf_log.process()
    # LinearWithVerification.perf_log.to_csv("perf.csv")

    print("======= Original Llama =======")
    ref_model(loader, model_path)

if __name__ == "__main__":
    main()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak memory allocated: {peak_memory:.2f} GB")

    #ref_model(model_path)
    #custom_model(model_path)
