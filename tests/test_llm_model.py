from verified_llm.llm_model import create_llm_model
import pytest 
from verified_llm.utils.log_utils import g_logger, logging
import torch

@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_verified_llm(model_path):
    stream_cpu = torch.cuda.Stream()
    stream_gpu = torch.cuda.Stream()
    verified_llm = create_llm_model(model_path, True, stream_cpu, stream_gpu)

    

    origin_llm = create_llm_model(model_path, False, stream_cpu, stream_gpu)

    

    
    


