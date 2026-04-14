import pytest
import torch

from verified_diffusers.zimage.config import VerifyConfig
from verified_llm.llm_model import create_llm_model


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_verified_llm(model_path):
    config = VerifyConfig(enabled=True, fail_on_error=True, profile_enabled=False)
    verified_llm = create_llm_model(model_path, verify=True, config=config)

    origin_llm = create_llm_model(model_path, verify=False)

    assert verified_llm._verify_runtime is not None
    assert origin_llm._verify_runtime is None
