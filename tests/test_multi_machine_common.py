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
