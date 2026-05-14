"""Unit tests for examples/_multi_machine_common.py."""
from __future__ import annotations

import importlib.util
import pathlib
import socket
import struct
import sys
import threading

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "examples" / "_multi_machine_common.py"


def _load_common():
    spec = importlib.util.spec_from_file_location("mmcommon", COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mmcommon"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_constants():
    m = _load_common()
    assert m.MSG_LOAD_REQ == 1
    assert m.MSG_LOAD_ACK == 2
    assert m.MSG_FORWARD_REQ == 3
    assert m.MSG_TENSOR == 4
    assert m.MSG_FORWARD_DONE == 5
    assert m.MSG_CLOSE == 6
    assert m.DTYPE_FP32 == 1
    assert m.DTYPE_FP16 == 2
    assert m._DTYPE_SIZE[m.DTYPE_FP32] == 4
    assert m._DTYPE_SIZE[m.DTYPE_FP16] == 2
    assert m._DTYPE_NAME[m.DTYPE_FP16] == "fp16"
    assert m._NAME_TO_DTYPE["fp32"] == m.DTYPE_FP32


def _pair_sockets():
    """Return a connected (client, server) socket pair via loopback."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    accepted: list = []

    def _accept():
        conn, _ = srv.accept()
        accepted.append(conn)

    t = threading.Thread(target=_accept)
    t.start()
    client.connect(("127.0.0.1", port))
    t.join(2.0)
    srv.close()
    return client, accepted[0]


def test_send_recv_msg_round_trip():
    m = _load_common()
    a, b = _pair_sockets()
    try:
        body = b"hello world"
        n = m.send_msg(a, m.MSG_LOAD_ACK, body)
        assert n == 8 + len(body)
        mt, got = m.recv_msg(b)
        assert mt == m.MSG_LOAD_ACK
        assert got == body
    finally:
        a.close(); b.close()


def test_recv_exactly_raises_on_eof():
    m = _load_common()
    a, b = _pair_sockets()
    try:
        a.close()
        with pytest.raises(ConnectionError):
            m.recv_exactly(b, 4)
    finally:
        b.close()


def test_pack_unpack_tensor_round_trip_fp32():
    m = _load_common()
    t = torch.randn(2, 3, 5, dtype=torch.float32)
    body = m.pack_tensor(request_id=42, op_tag=7, tensor=t,
                         wire_dtype_id=m.DTYPE_FP32)
    d = m.unpack_tensor(body)
    assert d["request_id"] == 42
    assert d["op_tag"] == 7
    assert d["dtype_id"] == m.DTYPE_FP32
    assert tuple(d["tensor"].shape) == (2, 3, 5)
    assert torch.equal(d["tensor"], t)


def test_pack_unpack_tensor_round_trip_fp16():
    m = _load_common()
    t = torch.randn(4, 8, dtype=torch.float32)  # source fp32
    body = m.pack_tensor(request_id=9, op_tag=300, tensor=t,
                         wire_dtype_id=m.DTYPE_FP16)
    d = m.unpack_tensor(body)
    assert d["op_tag"] == 300            # u16, must round-trip large values
    assert d["tensor"].dtype == torch.float16
    assert torch.allclose(d["tensor"].float(), t.half().float(), atol=0)


def test_pack_tensor_uses_uint16_op_tag():
    m = _load_common()
    t = torch.zeros(1, dtype=torch.float32)
    body = m.pack_tensor(request_id=0, op_tag=65535, tensor=t,
                         wire_dtype_id=m.DTYPE_FP32)
    # Header layout: <Q H B B  shape...>  → request_id(8) + op_tag(2) + dtype(1) + ndim(1) = 12 bytes
    op_tag = struct.unpack_from("<H", body, 8)[0]
    assert op_tag == 65535


def test_make_s_deterministic():
    m = _load_common()
    s1 = m.make_s(out_dim=16, k=10, seed=7)
    s2 = m.make_s(out_dim=16, k=10, seed=7)
    s3 = m.make_s(out_dim=16, k=10, seed=8)
    assert torch.equal(s1, s2)
    assert s1.shape == (16, 10)
    assert s1.dtype == torch.float32
    assert not torch.equal(s1, s3)


def test_slalom_verify_passes_for_correct_matmul():
    m = _load_common()
    torch.manual_seed(0)
    W = torch.randn(16, 8)
    x = torch.randn(2, 4, 8)
    y = x @ W.t()
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(W, s)
    assert s_tilde.shape == (8, 10)
    mse = m.slalom_verify(x, y, s, s_tilde)
    assert mse < 1e-10


def test_slalom_verify_safe_returns_inf_on_nan():
    m = _load_common()
    torch.manual_seed(0)
    W = torch.randn(16, 8)
    x = torch.randn(2, 4, 8)
    y = x @ W.t()
    y[0, 0, 0] = float("nan")
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(W, s)
    assert m.slalom_verify_safe(x, y, s, s_tilde) == float("inf")


def test_slalom_verify_fails_for_forged_y():
    m = _load_common()
    torch.manual_seed(0)
    W = torch.randn(16, 8)
    x = torch.randn(2, 4, 8)
    y_forged = torch.randn(2, 4, 16)  # not x @ W.T
    s = m.make_s(out_dim=16, k=10, seed=7)
    s_tilde = m.precompute_s_tilde(W, s)
    mse = m.slalom_verify(x, y_forged, s, s_tilde)
    assert mse > 1.0


def test_rope_llama_round_trip_against_hf():
    """Our apply_rope_llama matches HF apply_rotary_pos_emb."""
    m = _load_common()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    head_dim, max_seq = 64, 32
    cos, sin = m.precompute_rope_cos_sin(head_dim, max_seq, base=500000.0)
    assert cos.shape == (max_seq, head_dim)
    assert sin.shape == (max_seq, head_dim)

    torch.manual_seed(0)
    B, H, S, D = 2, 4, max_seq, head_dim
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    q_ours, k_ours = m.apply_rope_llama(q, k, cos, sin)

    # HF expects (cos, sin) shaped (1, S, D) or (B, S, D)
    cos_hf = cos.unsqueeze(0)
    sin_hf = sin.unsqueeze(0)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)
    assert torch.allclose(q_ours, q_hf, atol=1e-5)
    assert torch.allclose(k_ours, k_hf, atol=1e-5)


def test_rope_zimage_complex_cis_matches_reference():
    """Our apply_rotary_emb_zimage matches the verified_diffusers impl."""
    m = _load_common()
    from verified_diffusers.zimage.attention import apply_rotary_emb

    head_dim, max_seq = 64, 16
    freqs = m.precompute_zimage_freqs_cis(head_dim, max_seq, theta=10000.0)
    assert freqs.shape == (max_seq, head_dim // 2)
    assert freqs.dtype == torch.complex64 or freqs.dtype == torch.complex128

    torch.manual_seed(1)
    B, S, H, D = 2, max_seq, 4, head_dim
    x = torch.randn(B, S, H, D)
    out_ours = m.apply_rotary_emb_zimage(x, freqs)
    # Reference expects freqs shape (S, D/2) and applies it via complex math
    out_ref = apply_rotary_emb(x, freqs)
    assert torch.allclose(out_ours.float(), out_ref.float(), atol=1e-5)


def test_rmsnorm_cpu_matches_torch():
    m = _load_common()
    torch.manual_seed(2)
    x = torch.randn(2, 3, 16)
    weight = torch.randn(16)
    eps = 1e-6

    # No scale_offset → standard RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    out = m.rmsnorm_cpu(x, weight, eps, scale_offset=0.0)
    ref = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * weight
    assert torch.allclose(out, ref, atol=1e-6)

    # scale_offset=1.0 → Qwen3-style (1.0 + weight)
    out2 = m.rmsnorm_cpu(x, weight, eps, scale_offset=1.0)
    ref2 = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)
    assert torch.allclose(out2, ref2, atol=1e-6)

    # weight=None: identity scale
    out3 = m.rmsnorm_cpu(x, None, eps, scale_offset=0.0)
    ref3 = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    assert torch.allclose(out3, ref3, atol=1e-6)


def test_round_metrics_base_defaults():
    m = _load_common()
    rm = m.RoundMetricsBase(request_id=5)
    assert rm.request_id == 5
    assert rm.gpu_forward_t == 0.0
    assert rm.bytes_recv == 0
    assert rm.recv_tensors == {}
    assert rm.mse == {}
    assert rm.cpu_verify_per_op_t == {}
    assert rm.ok is True


def test_default_slalom_threshold_fp32():
    m = _load_common()
    assert m.default_slalom_threshold(m.DTYPE_FP32, in_dim=1024) == 1e-3
    assert m.default_slalom_threshold(m.DTYPE_FP32, in_dim=4096) == 1e-3


def test_default_slalom_threshold_fp16_scales_with_dim():
    m = _load_common()
    t_small = m.default_slalom_threshold(m.DTYPE_FP16, in_dim=128)
    t_large = m.default_slalom_threshold(m.DTYPE_FP16, in_dim=11008)
    assert t_small == max(1e-3, 128 * 2e-6)
    assert t_large == max(1e-3, 11008 * 2e-6)
    assert t_large > t_small


def test_default_slalom_threshold_custom_slope():
    m = _load_common()
    t = m.default_slalom_threshold(m.DTYPE_FP16, in_dim=4096, fp16_slope=4e-6)
    assert t == max(1e-3, 4096 * 4e-6)


def test_format_summary_basic_smoke():
    m = _load_common()
    rounds = [
        m.RoundMetricsBase(request_id=i, end_to_end_t=10.0, gpu_forward_t=3.0,
                           wire_recv_t=4.0, cpu_verify_t=2.0,
                           bytes_recv=1000, bytes_recv_predicted=1000,
                           mse={1: 1e-7, 2: 1e-7}, ok=True)
        for i in range(5)
    ]
    op_names = {1: "q", 2: "o"}
    s = m.format_summary(rounds, warmup=1, pipelined=False,
                         op_names=op_names,
                         config_lines=["foo: bar"],
                         link_gbps=None)
    assert "End-to-end" in s
    assert "rounds passed" in s
    assert "foo: bar" in s
    assert "q" in s and "o" in s
