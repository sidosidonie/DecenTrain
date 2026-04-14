import torch
from torch.nn import Linear
from pytest import mark

from verified_diffusers.zimage.config import VerifyConfig
from verified_diffusers.zimage.runtime import VerifyRuntime
from verified_llm.verify_linear import VerifyLinear, copy_to_cpu, freivalds_batch_matmul


@mark.parametrize("batch, hidden, inter, bias", [
    (32, 64, 128, False),
    (64, 128, 256, False),
    (128, 256, 512, False),
    (32, 64, 128, True),
    (64, 128, 256, True),
    (128, 256, 512, True),
])
def test_linear(batch, hidden, inter, bias):
    config = VerifyConfig(enabled=True, fail_on_error=True, profile_enabled=False)
    runtime = VerifyRuntime(config)

    origin_linear = Linear(hidden, inter, bias=bias).to("cuda")
    verify_linear = VerifyLinear(origin_linear, runtime, "test")

    x = torch.randn(batch, hidden, device="cuda", requires_grad=False)

    y_origin = origin_linear(x)
    y_v = verify_linear.forward(x)
    y_v_bias = verify_linear.add_bias(y_v)
    assert torch.allclose(y_origin, y_v_bias)

    runtime.flush()
    runtime.shutdown()


@mark.parametrize("batch, noise_scale", [
    (1, 1e-9),
    (2, 1e-9),
    (4, 1e-9),
    (8, 1e-9),
])
def test_mlp(batch, noise_scale):
    from transformers.models.llama.modeling_llama import LlamaMLP
    from transformers.models.llama.configuration_llama import LlamaConfig
    from verified_llm.mlp_layer import LlamaMLPVerify

    config = VerifyConfig(enabled=True, fail_on_error=True, profile_enabled=False)
    runtime = VerifyRuntime(config)

    llama_config = LlamaConfig("meta-llama/Llama-3.2-1B-Instruct")
    origin_mlp = LlamaMLP(llama_config).to("cuda")
    verify_mlp = LlamaMLPVerify(origin_mlp, runtime, noise_scale=noise_scale)

    x = torch.randn(batch, llama_config.hidden_size, device="cuda", requires_grad=False)
    y = origin_mlp.forward(x)
    y_v = verify_mlp.forward(x)
    assert torch.allclose(y, y_v)

    runtime.flush()
    runtime.shutdown()


@mark.parametrize("batch, seq_len, head_num, head_size", [
    (1, 128, 8, 64),
    (2, 64, 4, 32),
    (4, 32, 2, 64),
    (8, 16, 1, 128),
    (3, 32, 1, 64),
])
def test_freivalds_qk(batch, seq_len, head_num, head_size):
    stream = torch.cuda.Stream()
    q = torch.randn(batch, head_num, seq_len, head_size, device="cuda", requires_grad=False)
    k = torch.randn(batch, head_num, head_size, seq_len, device="cuda", requires_grad=False)
    qk = torch.matmul(q, k)
    assert qk.device.type == "cuda"

    q_cpu, _ = copy_to_cpu(q, stream)
    k_cpu, _ = copy_to_cpu(k, stream)
    qk_cpu, _ = copy_to_cpu(qk, stream)

    stream.synchronize()

    loss = freivalds_batch_matmul(q_cpu, k_cpu, qk_cpu)
    assert loss < 1e-8, f"Freivalds verification failed with loss {loss}"
