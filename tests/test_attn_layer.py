from verified_llm.attn_layer import *
from pytest import mark

@mark.parametrize("batch, seq_len, noise_scale", [
    (1, 64, 1e-7),
    (1, 128, 1e-7),
    (1, 256, 1e-7),
    (1, 64, None),
    (1, 128, None),
    (1, 256, None),
])
def test_attn(batch, seq_len, noise_scale):
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.Stream()
    config = LlamaConfig("meta-llama/Llama-3.2-1B-Instruct")
    origin_attn = LlamaAttention(config, layer_idx=0).to("cuda")
    verify_attn = LlamaAttentionVerify(origin_attn, cpu_stream, gpu_stream, noise_scale)

    def gen_attn_inputs():
        x = torch.randn(batch, seq_len, config.hidden_size, device="cuda", requires_grad=False)
        cos = torch.randn(batch, seq_len, config.head_dim, device="cuda", requires_grad=False)
        sin = torch.randn(batch, seq_len, config.head_dim, device="cuda", requires_grad=False)
        position_ids = torch.stack([cos, sin], dim=0)
        attention_mask = torch.ones_like(x, device="cuda", requires_grad=False)
        return x, position_ids, attention_mask

    x, position_ids, attention_mask = gen_attn_inputs()
    y = origin_attn.forward(hidden_states=x, position_embeddings=position_ids, attention_mask=None)
    y_v = verify_attn.forward(x, position_embeddings=position_ids, attention_mask=None)

    if noise_scale is None:
        assert torch.allclose(y[0], y_v[0])
        assert torch.allclose(y[1], y_v[1])
    else:
        assert torch.allclose(y[0], y_v[0], atol=1e-5)
        assert torch.allclose(y[1], y_v[1], atol=1e-5)

    #print(x)
    #print(y[0])
    #print(y_v[0])
    