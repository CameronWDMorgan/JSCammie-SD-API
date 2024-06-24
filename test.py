import diffusers
from diffusers import StableDiffusionPipeline
import torch
import PIL
from PIL import Image
import os
import time



print(f"Number of CUDA devices: {torch.cuda.device_count()}")

pipeline = diffusers.StableDiffusionXLPipeline.from_single_file('models/autismmixSDXL_autismmixLightning.safetensors', torch_dtype=torch.float16, variant="fp16")

# Ensure sampler uses "trailing" timesteps.
pipeline.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()

pipeline.enable_model_cpu_offload(gpu_id=1)


# image = pipeline(
#     prompt="((masterpiece, high quality, highres, emphasis lines)), 1girl, amy rose, denim jeans, white crop top, in city",
#     negative_prompt="nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
#     num_inference_steps=4,
#     guidance_scale=0,
#     num_images_per_prompt=1,
# ).images[0]

image = pipeline(
    prompt="score_9, score_8_up, score_7_up, flat color, source_cartoon, rating_suggestive, cel shading, highres, digital art, 2d, solo, Sonic \(series\), Amy Rose \(sonic\), big breasts, thick thighs, gray leggings, at beach, sunshine, lens flare",
    negative_prompt="score_1, score_2, score_3, score_4, (black and white, monochrome, hands, 3d, hyperrealistic, sfm)",
    width=768,
    height=1024,
    num_inference_steps=8,
    guidance_scale=0,
    num_images_per_prompt=1,
).images[0]

os.makedirs('outputs/test', exist_ok=True)

# Save the image, making sure it has unique filename:
image.save(f'outputs/test/{time.time()}.png')