import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.normalization import AdaLayerNormZero
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import FluxAttnProcessor2_0, Attention
from diffusers.utils import load_image
from verified_llm.verify_linear import SyncVerifyLinear, all_matmul
from diffusers import FluxPipeline, DiffusionPipeline, Flux2Pipeline
from pprint import pprint


torch.set_num_threads(8)
torch.set_num_interop_threads(2)

class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture (used in Flux).

    Parameters:
        dim (`int`): hidden dimension (e.g. 3072)
        num_attention_heads (`int`): number of heads (e.g. 24)
        attention_head_dim (`int`): dim per head (e.g. 128)
        qk_norm (`str`): type of q/k norm ("rms_norm" in Flux)
        eps (`float`): numerical epsilon
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        # --- adaptive layer norms (adaLN-Zero) ---
        # For dim=3072 this internally creates:
        #  - linear: 3072 -> 3072 * 6 = 18432 (matches your print)
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        # --- attention ---
        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "This block requires PyTorch with `scaled_dot_product_attention`."
            )

        # This is the "FluxAttention" you saw in the summary: it’s just a
        # specialized Attention with added_kv_proj_dim and FluxAttnProcessor2_0.
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,          # extra K/V for context stream
            dim_head=attention_head_dim,    # 128
            heads=num_attention_heads,      # 24
            out_dim=dim,                    # 3072
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        # --- MLPs for image and context streams ---
        # For dim=3072 with FeedForward default mult=4:
        #   internal dim = 4 * 3072 = 12288 (matches your print)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # diffusers keeps these for potential chunking; you can ignore them unless you
        # implement chunked forward.
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_emb=None,
        image_rotary_emb=None,
    ):
        """
        hidden_states:      [B, N_img, dim]
        encoder_hidden_states: [B, N_ctx, dim]  (text/context tokens)
        temb:              [B, dim] (time+text embedding)
        image_emb, image_rotary_emb: extra conditioning used by Flux attention
        """

        # --- adaLN-Zero for both streams ---
        # norm1 returns 5 chunks: gate_msa, shift_mlp, scale_mlp, gate_mlp
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # --- joint self-attention over image + context streams ---
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            image_emb=image_emb,
        )

        # --- image stream update ---
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # --- context stream update ---
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # Flux clamps only encoder_hidden_states (matches the official code)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # NOTE: ordering is (encoder_hidden_states, hidden_states), which is what
        # FluxTransformer2DModel expects when it loops over transformer_blocks.
        return encoder_hidden_states, hidden_states



def replace_linear(module: nn.Module):
    gpu_stream = torch.cuda.default_stream()
    cpu_stream = torch.cuda.Stream()
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = SyncVerifyLinear(
                child, gpu_stream, cpu_stream
            )

            # Copy weights
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)

            setattr(module, name, new_layer)

        else:
            replace_linear(child)

def test_sync_verify():
    import time
    l1 = nn.Linear(3072, 4096).to("cuda")

    sl1 = SyncVerifyLinear(l1, torch.cuda.default_stream(), torch.cuda.Stream())

    inp = torch.randn(3072, 3072).to("cuda")

    t1 = time.time()
    for i in range(20):
        l1_out = l1(inp)
    t2 = time.time()
    dur1 = (t2 - t1) / 20
    print("Linear time ", dur1)

    t1 = time.time()
    for i in range(20):
        sl1_out = sl1(inp)
    t2 = time.time()
    dur1 = (t2 - t1) / 20
    print("Verify Linear time ", dur1)
    

def replace_linear_for_pipeline(pipe : FluxPipeline):
    replace_linear(pipe.transformer)
    replace_linear(pipe.vae)
    print(pipe.vae)

def flux1():
    model_name = "black-forest-labs/FLUX.1-schnell"
    pipe = Flux2Pipeline.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to("cuda")
    

    prompt = "A cat holding a bottle, saying hello!"

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        replace_linear_for_pipeline(pipe)
        print(pipe)

        torch.cuda.synchronize()
        starter.record()

        image = pipe(
            prompt,
            guidance_scale=3.5,
            num_inference_steps=5
        ).images[0]

        ender.record()
        torch.cuda.synchronize()

        elapsed_ms = starter.elapsed_time(ender)
        print(f"Inference time: {elapsed_ms:.2f} ms")

        image.save("flux.png")


    pprint(all_matmul)

def flux2():
    model_name = "black-forest-labs/FLUX.2-klein-base-9B"
    pipe = Flux2Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to("cuda")
    

    prompt = "Turn this cat into a dog"
    input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")


    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        # replace_linear_for_pipeline(pipe)
        print(pipe)

        torch.cuda.synchronize()
        starter.record()

        print(input_image)
        print(type(prompt))
        
        image = pipe(image=input_image, prompt=prompt).images[0]

        ender.record()
        torch.cuda.synchronize()

        elapsed_ms = starter.elapsed_time(ender)
        print(f"Inference time: {elapsed_ms:.2f} ms")

        image.save("flux.png")


    pprint(all_matmul)

    # block = FluxTransformerBlock(
    #     dim=3072,
    #     num_attention_heads=24,
    #     attention_head_dim=128,
    #     qk_norm="rms_norm",
    #     eps=1e-6,
    # )

    # block = block.to("cuda")
    # replace_linear(block)
    


flux2()