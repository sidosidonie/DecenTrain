"""Shared helpers for the multi-machine examples.

Used by:
    examples/multi_machine_attn_llama.py
    examples/multi_machine_attn_zimage.py
    examples/multi_machine_zimage.py

The standalone FFN example (examples/multi_machine_ffn.py) does NOT
import from here — it stays self-contained as the canonical reference.
"""
from __future__ import annotations

import socket
import struct
from typing import Optional

import numpy as np
import torch


# ── Wire message types ──────────────────────────────────────────────
MSG_LOAD_REQ      = 1
MSG_LOAD_ACK      = 2
MSG_FORWARD_REQ   = 3
MSG_TENSOR        = 4   # generalized; carries (request_id, op_tag, dtype, shape, payload)
MSG_FORWARD_DONE  = 5
MSG_CLOSE         = 6


# ── Wire dtypes ─────────────────────────────────────────────────────
DTYPE_FP32 = 1
DTYPE_FP16 = 2

_TORCH_DTYPE = {DTYPE_FP32: torch.float32, DTYPE_FP16: torch.float16}
_NUMPY_DTYPE = {DTYPE_FP32: np.float32,    DTYPE_FP16: np.float16}
_DTYPE_SIZE  = {DTYPE_FP32: 4,             DTYPE_FP16: 2}
_DTYPE_NAME  = {DTYPE_FP32: "fp32",        DTYPE_FP16: "fp16"}
_NAME_TO_DTYPE = {v: k for k, v in _DTYPE_NAME.items()}


# ── Wire primitives ─────────────────────────────────────────────────
class WireProtocolError(RuntimeError):
    pass


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock or raise ConnectionError on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"unexpected EOF after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, msg_type: int, body: bytes) -> int:
    """Send a framed message. Returns total bytes written (header+body)."""
    header = struct.pack("<II", msg_type, len(body))
    sock.sendall(header + body)
    return len(header) + len(body)


def recv_msg(sock: socket.socket) -> tuple[int, bytes]:
    """Read one framed message. Returns (msg_type, body)."""
    header = recv_exactly(sock, 8)
    msg_type, body_len = struct.unpack("<II", header)
    body = recv_exactly(sock, body_len) if body_len else b""
    return msg_type, body


# ── Generic tensor packer ───────────────────────────────────────────
# Header layout: <Q H B B  shape[ndim] (I*ndim)>  + payload
#   request_id (u64) + op_tag (u16) + dtype_id (u8) + ndim (u8)
_TENSOR_HDR_FMT = "<QHBB"
_TENSOR_HDR_SIZE = struct.calcsize(_TENSOR_HDR_FMT)


def pack_tensor(request_id: int, op_tag: int, tensor: torch.Tensor,
                wire_dtype_id: int) -> bytes:
    np_dtype = _NUMPY_DTYPE[wire_dtype_id]
    torch_dtype = _TORCH_DTYPE[wire_dtype_id]
    t = tensor.detach().to(torch_dtype).contiguous().cpu()
    payload = t.numpy().astype(np_dtype, copy=False).tobytes()
    ndim = t.ndim
    shape_bytes = struct.pack(f"<{ndim}I", *t.shape)
    header = struct.pack(_TENSOR_HDR_FMT, request_id, op_tag,
                         wire_dtype_id, ndim)
    return header + shape_bytes + payload


def unpack_tensor(body: bytes) -> dict:
    request_id, op_tag, dtype_id, ndim = struct.unpack_from(
        _TENSOR_HDR_FMT, body, 0)
    off = _TENSOR_HDR_SIZE
    shape = struct.unpack_from(f"<{ndim}I", body, off)
    off += 4 * ndim
    payload = body[off:]
    np_dtype = _NUMPY_DTYPE[dtype_id]
    torch_dtype = _TORCH_DTYPE[dtype_id]
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(arr.copy()).to(torch_dtype)
    return {"request_id": request_id, "op_tag": op_tag,
            "dtype_id": dtype_id, "tensor": tensor}
