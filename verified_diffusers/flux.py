import torch
from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A cat holding a sign that says hello world"
# image = pipe(
#     prompt,
#     height=1024,
#     width=1024,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cuda").manual_seed(0)
# ).images[0]
# image.save("flux-dev.png")


import torch
from diffusers import FluxPipeline


torch.set_num_threads(16)


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-schnell.png")

