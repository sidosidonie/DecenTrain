from verified_training.llm_model import create_llm_model
import pytest 
from verified_training.utils.log_utils import g_logger, logging

@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_verified_llm(model_path):
    v_llm = create_llm_model(model_path, True)
    print(v_llm)

if __name__ == "__main__":
    g_logger.setLevel(level=logging.DEBUG)
    test_verified_llm("meta-llama/Llama-3.2-1B-Instruct")

