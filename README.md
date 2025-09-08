
## Verified Large Language Model 

Reimplemented LlamaMLP and LlamaAttention layers to support freivalds verification.

## Usage

### Install

In conda or virtual env, install dependencies.

```bash
pip install -r requirements.txt
```

### Dataset

c4 test dataset, located in dataset/c4, this is called by verified_llm/utils/dataset_utils.py

### Test noise  
```
mkdir logs
chmod +x ppl.sh
./ppl.sh
```
The outputs are in logs/ppl-llama-noise-${noise_scale}-limit-${limit_samples}-loss

### Test for end-to-end 

```
python verified_llm/eval.py
```

## Code Structure

- verified_llm: contains all code
- verified_llm/verified_linear.py: reimplement Linear as VerifiedLinear, implement frevalds algorithms.
- verified_llm/attn_layer.py: reimplement LlamaAttention as LlamaAttentionVerify
- verified_llm/mlp_layer.py: reimplement LlamaMLP as LlamaMLPVerify
- verified_llm/eval.py: the main entry functions
- utils/: helpers like dataset functions
- legacy/: old code that is not useful

