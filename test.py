import diffusers
from diffusers import StableDiffusionPipeline
import torch
import PIL
from PIL import Image

import DeepCache

from DeepCache import DeepCacheSDHelper

import tomesd
import hidiffusion

import onediffx
from onediffx import compile_pipe

print(f"Number of CUDA devices: {torch.cuda.device_count()}")

pipeline = StableDiffusionPipeline.from_single_file(
    './models/fluffysonic_v20.safetensors',
    # torch_dtype=torch.float16,
)

pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()

# image = pipeline(
#     prompt="((masterpiece, high quality, highres, emphasis lines)), 1girl, amy rose, denim jeans, white crop top, in city",
#     negative_prompt="nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
#     num_inference_steps=15,
#     num_images_per_prompt=1,
# ).images[0]

# image.save('PLEASEWORK_output.png')

helper = DeepCacheSDHelper(pipe=pipeline)
helper.set_params(
    cache_interval=3,
    cache_branch_id=0,
)
helper.enable()
    
tomesd.apply_patch(pipeline, ratio=0.2)

compile_pipe(pipeline)

image = pipeline(
    prompt="((masterpiece, high quality, highres, emphasis lines)), 1girl, amy rose, denim jeans, white crop top, in city",
    negative_prompt="nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    num_inference_steps=15,
    num_images_per_prompt=1,
).images[0]

image.save('output-1.png')

image = pipeline(
    prompt="((masterpiece, high quality, highres, emphasis lines)), 1girl, amy rose, denim jeans, white crop top, in city",
    negative_prompt="nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    num_inference_steps=15,
    num_images_per_prompt=1,
).images[0]

image.save('output-2.png')
