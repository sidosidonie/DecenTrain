"""Functional tests for examples/multi_machine_ffn.py."""
from __future__ import annotations

import importlib.util
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "multi_machine_ffn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mmffn", EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_and_exposes_constants():
    m = _load_module()
    assert m.MSG_LOAD_REQ == 1
    assert m.MSG_LOAD_ACK == 2
    assert m.MSG_FORWARD_REQ == 3
    assert m.MSG_ACTIVATION == 4
    assert m.MSG_FORWARD_DONE == 5
    assert m.MSG_CLOSE == 6
    assert m.OP_W1 == 1
    assert m.OP_W3 == 2
    assert m.OP_W2 == 3
    assert m.DTYPE_FP32 == 1
    assert m.DTYPE_FP16 == 2
