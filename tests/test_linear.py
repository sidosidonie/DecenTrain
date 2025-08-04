from verified_llm.mlp_layer import *
from pytest import mark

@mark.parametrize("batch, hidden, inter, bias", [
    (32, 64, 128, False),
    (64, 128, 256, False),
    (128, 256, 512, False),
    (32, 64, 128, True),
    (64, 128, 256, True),
    (128, 256, 512, True),
])
def test_linear(batch, hidden, inter, bias):
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.Stream()
    origin_linear = Linear(hidden, inter, bias=bias).to("cuda")
    verify_linear = VerifyLinear(origin_linear, cpu_stream, gpu_stream)

    x = torch.randn(batch, hidden, device="cuda", requires_grad=False)

    y = origin_linear(x)
    y_v = verify_linear.forward(x)
    y_v_bias = verify_linear.add_bias(y_v)
    assert torch.allclose(y, y_v_bias)

    loss, y_cpu = verify_linear.verify_forward(x, y_v)

    y_cpu_bias = verify_linear.add_bias_cpu(y_cpu)
    assert torch.allclose(y.to("cpu"), y_cpu_bias)
    
    assert loss < 1e-5, f"Verification failed with loss {loss}"


@mark.parametrize("batch", [1, 2, 4, 8])
def test_mlp(batch):
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.Stream()
    config = LlamaConfig("meta-llama/Llama-3.2-1B-Instruct")
    origin_mlp = LlamaMLP(config).to("cuda")
    verify_mlp = LlamaMLPVerify(origin_mlp, cpu_stream, gpu_stream)

    x = torch.randn(batch, config.hidden_size, device="cuda", requires_grad=False)
    y = origin_mlp.forward(x)
    y_v = verify_mlp.forward(x)
    assert torch.allclose(y, y_v)
