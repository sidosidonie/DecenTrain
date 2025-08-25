from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
from pprint import pprint
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import *
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Callable, Optional, Tuple, OrderedDict
from transformers.activations import ACT2FN
from torch import Tensor
import math
import time
from verified_llm.verify_linear import freivalds_algorithm, freivalds_batch_matmul
from torch.nn import functional as F, init
from torch.nn import Linear, Module, Parameter
from torch.autograd import Function
import torch
import sys
import os
from verified_llm.utils.log_utils import g_logger
from verified_llm.utils.profiler import Profiler 
from verified_llm.verify_linear import VerifyLinear, copy_to_cpu, add_noise

DISABLE_VERIFY = False

def get_sparse_tensor_cap(t):
    values_mem = t.values().element_size() * t.values().numel()
    indices_mem = t.crow_indices().element_size() * t.crow_indices().numel()
    col_indices_mem = t.col_indices().element_size() * t.col_indices().numel()
    total_mem = values_mem + indices_mem + col_indices_mem
    print(f"Estimated sparse tensor memory: {total_mem / 1024:.2f} KB")
    print("Estimated dense tensor memory: {:.2f} KB".format(t.shape.numel() * t.element_size() / 1024))
    return total_mem

def gen_mask(attn_mask):
    if attn_mask is None:
        return attn_mask
    x = attn_mask
    x = torch.where(x == 0, torch.tensor(1., device=x.device, dtype=x.dtype), x)
    x = torch.where(torch.isneginf(x), torch.tensor(0., device=x.device, dtype=x.dtype), x)
    return x

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
    #prof = kwargs["prof"] if "prof" in kwargs else None
    #before_attn = kwargs["before_attn"] if "before_attn" in kwargs else None

    g_logger.info("parallel: block1")

    # t2 = prof.new("attn-qkv:  softmax, dropout, kv | qk, softmax, dropout")
    # t2_cpu = prof.new(">> attn-kv-cpu")
    # t2_gpu = prof.new(">> attn-kv-gpu")

    # t2_0 = prof.new(">>>> attn-kv-cpu-copy_attn")
    # t2_1 = prof.new(">>>> attn-kv-cpu-verify")
    # t2_2 = prof.new(">>>> attn-kv-cpu-matmul_qk")
    # t2_3 = prof.new(">>>> attn-kv-cpu-scale-softmax-dropout")


    # t3 = prof.new("kv: o_proj | kv_verify")
    # t4 = prof.new("o_proj_verify")

    noise_scale = kwargs["noise_scale"] if "noise_scale" in kwargs else None
    sliding_window_size = kwargs["sliding_window_size"] if "sliding_window_size" in kwargs else None

    if not DISABLE_VERIFY:
        with torch.cuda.stream(st):
            key_states_cpu = repeat_kv(key_cpu, module.num_key_value_groups)
            key_states_cpu = key_states_cpu.transpose_(2, 3)
            value_states_cpu = repeat_kv(value_cpu, module.num_key_value_groups)

    # key_rep, key_rep_cpu
    key_states = repeat_kv(key, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3))
    if not DISABLE_VERIFY:
        attn_weights = add_noise(attn_weights, noise_scale)

    mask01 = gen_mask(attention_mask)

    attn_weights = attn_weights * mask01

    torch.cuda.synchronize()  # sync gpu attn
    # before_attn.ed()

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights_mask = attn_weights + causal_mask
    else:
        causal_mask = None
        attn_weights_mask = attn_weights

    # ================================
    # t2.st()
    if not DISABLE_VERIFY:
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e1.record()
        attn_weights_sparse = attn_weights.to_sparse_csr()  # or .to_sparse() for COO format
        e2.synchronize()
        e2.record()
        g_logger.info(f"====== Sparse convert time: {e1.elapsed_time(e2)} ms")

        get_sparse_tensor_cap(attn_weights_sparse)

        with torch.cuda.stream(st):
            # t2_cpu.st()
            # t2_0.st()
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e1.record()
            attn_weights_cpu, _ = copy_to_cpu(attn_weights_sparse, st)
            #attn_weights_cpu, _ = copy_to_cpu(attn_weights, st)
            st.synchronize()
            e2.record()
            g_logger.info(f"====== Copy to cpu time: {e1.elapsed_time(e2)} ms")

            attn_weights_cpu = attn_weights_cpu.to_dense()
            #cur_seq_len = key_states_cpu.shape[-1]
            #attn_weights_current_cpu = attn_weights_cpu[:, :, :, 0:cur_seq_len]

            g_logger.info(f"qk verify shapes: {query_cpu.shape=}, {key_states_cpu.shape=}, {attn_weights_cpu.shape=}")
            loss = freivalds_batch_matmul(query_cpu, key_states_cpu, attn_weights_cpu)
            g_logger.info(f"QK verify loss: {loss}")
            st.synchronize()
            # t2_1.ed()

            attn_weights_scale_cpu = attn_weights_cpu * scaling
            if attention_mask is not None:
                attn_weights_scale_cpu = attn_weights_scale_cpu + causal_mask.cpu()

            attn_weights_scale_soft_cpu = F.softmax(attn_weights_scale_cpu, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights_drop_cpu = F.dropout(attn_weights_scale_soft_cpu, p=dropout, training=module.training)
            st.synchronize()
            # t2_3.ed()
            # t2_cpu.ed()

    torch.cuda.synchronize()  # sync gpu attn
    # t2_gpu.st()
    attn_weights_scale = attn_weights_mask * scaling
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights_scale_soft = F.softmax(attn_weights_scale, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights_drop = F.dropout(attn_weights_scale_soft, p=dropout, training=module.training)
    qkv = torch.matmul(attn_weights_drop, value_states)
    if not DISABLE_VERIFY:
        qkv = add_noise(qkv, noise_scale)
    torch.cuda.synchronize()
    #t2_gpu.ed()

    #t2.ed()
    g_logger.info("parallel: block3")
    # verify key state

    #t3.st()
    final_out = qkv.transpose(1, 2).contiguous()
    attn_output = final_out.reshape(*input_shape, -1).contiguous()
    output = module.o_proj.forward(attn_output)

    if not DISABLE_VERIFY:
        with torch.cuda.stream(st):
            g_logger.info(f"attn drop shape {attn_weights_drop.shape}")
            torch.cuda.synchronize()
            final_out_cpu, _copy_e = copy_to_cpu(qkv, st)
            #attn_weights_drop_cpu_cur = attn_weights_drop_cpu[:, :, :, 0:cur_seq_len]
            #loss = freivalds_algorithm(attn_weights_drop_cpu_cur, value_states_cpu, final_out_cpu, st, e3=_copy_e)
            st.synchronize()
            loss = freivalds_batch_matmul(attn_weights_drop_cpu, value_states_cpu, final_out_cpu)
            g_logger.info(f"KV Verify loss - {loss}")

    torch.cuda.synchronize()
    #t3.ed()

    #t4.st()
    if not DISABLE_VERIFY:
        o_loss, out_cpu = module.o_proj.verify_forward(attn_output, output)
        g_logger.info(f"O Verify loss: {o_loss}")
    #t4.ed()

    return output, attn_weights_drop

class LlamaAttentionVerify(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, llama : LlamaAttention, st_cpu, st_gpu, noise=None, sliding_window_size=None):
        super().__init__()
        self.config = llama.config
        self.layer_idx = llama.layer_idx
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True


        self.q_proj = VerifyLinear(llama.q_proj, st_cpu, st_gpu)
        self.k_proj = VerifyLinear(llama.k_proj, st_cpu, st_gpu)
        self.v_proj = VerifyLinear(llama.v_proj, st_cpu, st_gpu)
        self.o_proj = VerifyLinear(llama.o_proj, st_cpu, st_gpu)

        self.stream_cpu = st_cpu
        self.stream_gpu = st_gpu
        self.noise = noise
        self.sliding_window_size = sliding_window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        g_logger.info(f"===== VerifyLlamaAttention forward for layer {self.layer_idx}")

        #q_proj_time = self.duration.new("q_proj: q_proj")
        #k_proj_time = self.duration.new("k_proj: k_proj | q_proj_verify")
        #v_proj_time = self.duration.new("v_proj: v_proj | k_verify") 
        #before_attn = self.duration.new("before_attn: rotary, rep_kv, qk | v_verify, rotary, rep_kv")

        #inputs_cpu, event_copy_input = copy_to_cpu(inputs_gpu, self.stream)
        #hidden_states, position = inputs_gpu

        cos, sin = position_embeddings[0], position_embeddings[1]
        hidden_states_cpu, event_hidden = copy_to_cpu(hidden_states, self.stream_cpu)
        cos_cpu, e_cos = copy_to_cpu(cos, self.stream_cpu)
        sin_cpu, e_sin = copy_to_cpu(sin, self.stream_cpu)
        e_cos.synchronize()
        e_sin.synchronize()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        #q_proj_time.st()
        query_states1 = self.q_proj.forward(hidden_states)
        #q_proj_time.ed()

        #k_proj_time.st()
        key_states1 = self.k_proj.forward(hidden_states)
        self.stream_cpu.synchronize()
        self.stream_gpu.synchronize()
        if not DISABLE_VERIFY:
            event_hidden.synchronize()
            with torch.cuda.stream(self.stream_cpu):
                q_loss, query_states_cpu = self.q_proj.verify_forward(hidden_states_cpu, query_states1)
                g_logger.info(f"Q Verify loss: {q_loss}")
                query_states_cpu = self.q_proj.add_bias_cpu(query_states_cpu)
                query_states_cpu = query_states_cpu.view(hidden_shape).transpose(1, 2)

        query_states1 = self.q_proj.add_bias(query_states1)
        query_states = query_states1.view(hidden_shape).transpose(1, 2)
        torch.cuda.synchronize()
        #k_proj_time.ed()

        # k_verify | v_proj
        #v_proj_time.st()
        value_states1 = self.v_proj.forward(hidden_states)
        if not DISABLE_VERIFY:
            with torch.cuda.stream(self.stream_cpu):
                k_loss, key_states_cpu = self.k_proj.verify_forward(hidden_states_cpu, key_states1)
                g_logger.info(f"K Verify loss: {k_loss}")
                key_states_cpu = self.k_proj.add_bias_cpu(key_states_cpu)
                key_states_cpu = key_states_cpu.view(hidden_shape).transpose(1, 2)

        key_states1 = self.k_proj.add_bias(key_states1)
        key_states = key_states1.view(hidden_shape).transpose(1, 2)
        torch.cuda.synchronize()
        #v_proj_time.ed()


        # v_verify | qxk
        #before_attn.st()

        value_states = value_states1.view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if not DISABLE_VERIFY:
            with torch.cuda.stream(self.stream_cpu):
                v_loss, value_states_cpu = self.v_proj.verify_forward(hidden_states_cpu, value_states1)
                g_logger.info(f"V Verify loss: {v_loss}")
                value_states_cpu = self.v_proj.add_bias_cpu(value_states_cpu)
                value_states_cpu = value_states_cpu.view(hidden_shape).transpose(1, 2)
                query_states_cpu, key_states_cpu = apply_rotary_pos_emb(query_states_cpu, key_states_cpu, cos_cpu, sin_cpu, unsqueeze_dim=1)

        value_states1 = self.v_proj.add_bias(value_states1)

        if DISABLE_VERIFY:
            query_states_cpu = None
            key_states_cpu = None
            value_states_cpu = None


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            if not DISABLE_VERIFY:
                key_states_cpu, _e1 = copy_to_cpu(key_states, self.stream_cpu)
                value_states_cpu, _e2 = copy_to_cpu(value_states, self.stream_cpu)
                _e1.synchronize()
                _e2.synchronize()

        attn_output, attn_weights = eager_attention_forward_verify(
            self.stream_cpu,
            self,
            query_states,
            key_states,
            value_states,
            query_states_cpu,
            key_states_cpu,
            value_states_cpu,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            input_shape=input_shape,
            before_attn = None,
            noise_scale=self.noise,
            sliding_window_size=self.sliding_window_size
        )

        return attn_output, attn_weights

def sliding_window_mask(batch, head_dim, seq_len, window, device="cpu"):
    # indices [L, L]
    arange = torch.arange(seq_len, device=device)
    diff = arange[:, None] - arange[None, :] 
    # allow positions 0..window (causal + window)
    mask = torch.where((diff >= 0) & (diff <= window), 0., float('-inf'))
    # mask: [seq_len, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    mask = mask.expand(batch, head_dim, seq_len, seq_len)  # [batch, head_dim, seq_len, seq_len]
    return mask

def test_attn(batch, seq_len, noise_scale, sliding_window_size):
    cpu_stream = torch.cuda.Stream()
    gpu_stream = torch.cuda.Stream()
    config = LlamaConfig("meta-llama/Llama-3.2-1B-Instruct")
    config.hidden_size = 10
    config.head_dim = 32
    origin_attn = LlamaAttention(config, layer_idx=0).to("cuda")
    verify_attn = LlamaAttentionVerify(origin_attn, cpu_stream, gpu_stream, noise_scale, sliding_window_size)

    def gen_attn_inputs():
        x = torch.randn(batch, seq_len, config.hidden_size, device="cuda", requires_grad=False)
        cos = torch.randn(batch, seq_len, config.head_dim, device="cuda", requires_grad=False)
        sin = torch.randn(batch, seq_len, config.head_dim, device="cuda", requires_grad=False)
        position_ids = torch.stack([cos, sin], dim=0)
        attention_mask = sliding_window_mask(batch, config.head_dim, seq_len, sliding_window_size, device="cuda")
        return x, position_ids, attention_mask

    x, position_ids, attention_mask = gen_attn_inputs()
    #y = origin_attn.forward(hidden_states=x, position_embeddings=position_ids, attention_mask=attention_mask)
    #exit(-1)
    y_v = verify_attn.forward(x, position_embeddings=position_ids, attention_mask=attention_mask)

    #if noise_scale is None:
    #    assert torch.allclose(y[0], y_v[0])
    #    assert torch.allclose(y[1], y_v[1])
    #else:
    #    assert torch.allclose(y[0], y_v[0], atol=1e-5)
    #    assert torch.allclose(y[1], y_v[1], atol=1e-5)

if __name__ == "__main__":
    test_attn(1, 1024, None, 256)