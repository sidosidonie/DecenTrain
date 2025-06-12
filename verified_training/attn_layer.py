from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pprint import pprint
from transformers.models.llama.modeling_llama import *
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Callable, Optional, Tuple, OrderedDict
from transformers.activations import ACT2FN
from torch import Tensor
import math
import time
from verified_training.verification import time_profile, freivalds_algorithm, freivalds_algorithm_linear
from torch.nn import functional as F, init
from torch.nn import Linear, Module, Parameter
from torch.autograd import Function
import torch
import sys
import os
from verified_training.utils.log_utils import g_logger
from verified_training.utils.profiler import Profiler 
from verified_training.mlp_layer import copy_to_cpu, all_close, VerifyLinear

DISABLE_VERIFY = False

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward_verify(
    st : torch.cuda.Stream ,
    module: Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_cpu: torch.Tensor,
    key_cpu: torch.Tensor,
    value_cpu: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    input_shape = None,
    **kwargs,
):
    prof = kwargs["prof"] if "prof" in kwargs else None
    before_attn = kwargs["before_attn"] if "before_attn" in kwargs else None

    g_logger.info("parallel: block1")

    t2 = prof.new("attn-qkv:  softmax, dropout, kv | qk, softmax, dropout")
    t2_cpu = prof.new(">> attn-kv-cpu")
    t2_gpu = prof.new(">> attn-kv-gpu")

    t2_0 = prof.new(">>>> attn-kv-cpu-copy_attn")
    t2_1 = prof.new(">>>> attn-kv-cpu-verify")
    t2_2 = prof.new(">>>> attn-kv-cpu-matmul_qk")
    t2_3 = prof.new(">>>> attn-kv-cpu-scale-softmax-dropout")


    t3 = prof.new("kv: o_proj | kv_verify")
    t4 = prof.new("o_proj_verify")

    if not DISABLE_VERIFY:
        with torch.cuda.stream(st):
            key_states_cpu = repeat_kv(key_cpu, module.num_key_value_groups)
            key_states_cpu = key_states_cpu.transpose_(2, 3)
            value_states_cpu = repeat_kv(value_cpu, module.num_key_value_groups)

    # key_rep, key_rep_cpu
    key_states = repeat_kv(key, module.num_key_value_groups)
    g_logger.info(f"key_states shape {key_states.shape}")
    g_logger.info(f"query shape {query.shape}")
    attn_weights = torch.matmul(query, key_states.transpose(2, 3))
    g_logger.info(f"attn shape {attn_weights.shape}")
    torch.cuda.synchronize()  # sync gpu attn
    before_attn.ed()

    # ================================
    t2.st()
    if not DISABLE_VERIFY:
        with torch.cuda.stream(st):
            t2_cpu.st()
            t2_0.st()
            attn_weights_cpu = copy_to_cpu(attn_weights, st)
            st.synchronize()
            t2_0.ed()

            #t2_2.st()
            #attn_weights_cpu = torch.matmul(query_cpu, key_states_cpu)
            #st.synchronize()
            #t2_2.ed()

            t2_1.st()
            loss = freivalds_algorithm(query_cpu, key_states_cpu, attn_weights_cpu)
            st.synchronize()
            t2_1.ed()


            t2_3.st()
            attn_weights_scale_cpu = attn_weights_cpu * scaling
            attn_weights_scale_soft_cpu = F.softmax(attn_weights_scale_cpu, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights_drop_cpu = F.dropout(attn_weights_scale_soft_cpu, p=dropout, training=module.training)
            st.synchronize()
            t2_3.ed()
            t2_cpu.ed()

    torch.cuda.synchronize()  # sync gpu attn
    t2_gpu.st()
    attn_weights_scale = attn_weights * scaling
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights_scale_soft = F.softmax(attn_weights_scale, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights_drop = F.dropout(attn_weights_scale_soft, p=dropout, training=module.training)
    qkv = torch.matmul(attn_weights_drop, value_states)
    torch.cuda.synchronize()
    t2_gpu.ed()

    t2.ed()
    g_logger.info("parallel: block3")
    # verify key state

    # if attention_mask is not None:
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask

    t3.st()
    final_out = qkv.transpose(1, 2).contiguous()
    attn_output = final_out.reshape(*input_shape, -1).contiguous()
    output = module.o_proj.forward(attn_output)

    if not DISABLE_VERIFY:
        with torch.cuda.stream(st):
            g_logger.info(f"attn drop shape {attn_weights_drop.shape}")
            final_out_cpu = copy_to_cpu(qkv, st)
            st.synchronize()
            loss = freivalds_algorithm(attn_weights_drop_cpu, value_states_cpu, final_out_cpu)

    torch.cuda.synchronize()
    t3.ed()

    t4.st()
    if not DISABLE_VERIFY:
        module.o_proj.verify_forward(attn_output, output)
    t4.ed()

    return output, None

class VerifyLlamaAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, llama : LlamaAttention, st, dur):
        super().__init__()
        self.original_module = llama
        config = llama.config
        self.config = llama.config
        self.layer_idx = llama.layer_idx
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = VerifyLinear(llama.q_proj, st)
        self.k_proj = VerifyLinear(llama.k_proj, st)
        self.v_proj = VerifyLinear(llama.v_proj, st)
        self.o_proj = VerifyLinear(llama.o_proj, st)

        self.stream = st
        self.duration = dur
        
    def forward(
        self,
        inputs_gpu: Tuple[torch.Tensor, torch.Tensor],
        inputs_cpu: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        q_proj_time = self.duration.new("q_proj: q_proj")
        k_proj_time = self.duration.new("k_proj: k_proj | q_proj_verify")
        v_proj_time = self.duration.new("v_proj: v_proj | k_verify") 
        before_attn = self.duration.new("before_attn: rotary, rep_kv, qk | v_verify, rotary, rep_kv")

        hidden_states, position = inputs_gpu
        cos, sin = position[0], position[1]
        hidden_states_cpu, position_cpu = inputs_cpu
        cos_cpu, sin_cpu = position_cpu[0], position_cpu[1]

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        g_logger.info("Query Proj")
        torch.cuda.synchronize()

        g_logger.info(f"hidden_states shape {hidden_states.shape}")
        g_logger.info(f"hidden_states weight shape {self.q_proj.weight.shape}")

        q_proj_time.st()
        query_states1 = self.q_proj.forward(hidden_states)
        torch.cuda.synchronize()
        q_proj_time.ed()

        g_logger.info("query out shape {}".format(query_states1.shape))

        k_proj_time.st()
        key_states1 = self.k_proj.forward(hidden_states)
        if not DISABLE_VERIFY:
            with torch.cuda.stream(self.stream):
                self.stream.synchronize()
                query_states1_cpy = copy_to_cpu(query_states1, self.stream)
                self.stream.synchronize()
                query_states_cpu = self.q_proj.verify_forward_mm(hidden_states_cpu, query_states1_cpy)
                query_states_cpu = query_states_cpu.view(hidden_shape).transpose(1, 2)

        query_states = query_states1.view(hidden_shape).transpose(1, 2)
        torch.cuda.synchronize()
        k_proj_time.ed()

        # k_verify | v_proj
        v_proj_time.st()
        value_states1 = self.v_proj.forward(hidden_states)
        if not DISABLE_VERIFY:
            with torch.cuda.stream(self.stream):
                key_states_cpu = self.k_proj.verify_forward_mm(hidden_states_cpu, key_states1)
                key_states_cpu = key_states_cpu.view(hidden_shape).transpose(1, 2)

        key_states = key_states1.view(hidden_shape).transpose(1, 2)
        torch.cuda.synchronize()
        v_proj_time.ed()


        # v_verify | qxk
        before_attn.st()

        value_states = value_states1.view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if not DISABLE_VERIFY:
            with torch.cuda.stream(self.stream):
                value_states_cpu = self.v_proj.verify_forward_mm(hidden_states_cpu, value_states1)
                value_states_cpu = value_states_cpu.view(hidden_shape).transpose(1, 2)
                query_states_cpu, key_states_cpu = apply_rotary_pos_emb(query_states_cpu, key_states_cpu, cos_cpu, sin_cpu, unsqueeze_dim=1)

        if DISABLE_VERIFY:
            query_states_cpu = None
            key_states_cpu = None
            value_states_cpu = None

        attn_output, attn_output_cpu = eager_attention_forward_verify(
            self.stream,
            self,
            query_states,
            key_states,
            value_states,
            query_states_cpu,
            key_states_cpu,
            value_states_cpu,
            None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            input_shape=input_shape,
            prof=self.duration,
            before_attn = before_attn
        )

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # output = self.o_proj.forward(attn_output)
        # torch.cuda.synchronize()
        # self.o_proj.verify_forward(attn_output, output)
        g_logger.info(f"{attn_output.shape=}")
        exit(-1)
        return attn_output 

def make_attn(attn : LlamaAttention, batch, seq_len):
    batch_size = batch
    seq_length = seq_len
    hidden_size = attn.config.hidden_size  # varies by model size
    num_heads = attn.config.num_attention_heads  # e.g., for LLaMA-7B
    head_dim = hidden_size // num_heads  # e.g., 128 // 16 = 8

    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    cos = torch.randn(batch_size, seq_length, head_dim)
    sin = torch.randn(batch_size, seq_length, head_dim)
    position_ids = torch.stack([cos, sin], dim=0)
    return hidden_states, position_ids

def llama_attn_test(prof : Profiler, batch, seq_len):
    ## test original
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config = LlamaConfig(model_name)
    #config.hidden_size = 128
    config.mlp_bias = False
    origin_attn_model = LlamaAttention(config, 0)
    inputs = make_attn(origin_attn_model, batch, seq_len)
    st = torch.cuda.default_stream(device=torch.device("cuda"))

    inputs_gpu = [i.to("cuda") for i in inputs if i is not None ]
    t1 = prof.add_time_span("FullAttn")
    t2 = prof.add_time_span("Breakdown")

    v_attn = VerifyLlamaAttention(origin_attn_model, st, t2)
    
    origin_attn_model.to("cuda")
    v_attn.to("cuda")
    
    for _ in range(prof.iter_n):
        origin_t = t1.new("Origin");
        verify_t = t1.new("Verify")
        origin_t2 = t2.new("Origin");
        verify_t2 = t2.new("Verify")
        with torch.enable_grad():
            origin_t.st()
            origin_t2.st()
            out = origin_attn_model(*inputs_gpu, attention_mask=None)
            torch.cuda.synchronize()
            origin_t.ed()
            origin_t2.ed()

            verify_t.st()
            verify_t2.st()
            out_v = v_attn(inputs_gpu, inputs)
            torch.cuda.synchronize()
            verify_t.ed()
            verify_t2.ed()

        t1.new_iter()
        t2.new_iter()

        g_logger.info(f"Result close: {all_close(out[0], out_v[0])}")
        g_logger.info(f"Result close: {all_close(out[-1], out_v[-1])}")

    print(f"{DISABLE_VERIFY=}")
    pprint(prof.report())

if __name__ == "__main__":
    prof = Profiler()
    DISABLE_VERIFY = False
    llama_attn_test(prof, 1, 2048)
    prof1 = Profiler()
    DISABLE_VERIFY = True
    llama_attn_test(prof1, 1, 2048)

    origin = prof.report()["Breakdown"]
    verify = prof1.report()["Breakdown"]

    pprint(origin)
    pprint(verify)

    l = []
    for k, vv in verify.items():
        v = origin[k]
        if vv == 0:
            overhead = 0
        else:
            overhead = v / vv
        instance = (k, v, vv, overhead, v - vv)
        l.append(instance)

    pprint(l)
    pp = pd.DataFrame(l, columns=["name", "verify", "origin", "overhead", "diff"])
    print(pp)
    pp.to_csv("attn_verify2.csv")
    m = pp.to_markdown(index=False)
    print(m)
