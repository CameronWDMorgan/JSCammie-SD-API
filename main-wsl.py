import base64
from flask import Flask, request, jsonify, send_file
import logging
from flask_cors import CORS
import io
import random
import json
import threading
import imageio
import yaml
import uuid
import torch
import tomesd
from diffusers import AutoencoderKL, AutoPipelineForInpainting, AutoPipelineForImage2Image, AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, AnimateDiffPipeline, MotionAdapter, ControlNetModel, StableDiffusionUpscalePipeline, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline
from diffusers.utils import load_image, export_to_gif
from diffusers.models.attention_processor import AttnProcessor2_0
import time
from io import BytesIO
# from transformers import AutoTokenizer, AutoModelForCausalLM

from torchvision import transforms


from moviepy.editor import ImageSequenceClip


from concurrent.futures import ThreadPoolExecutor
import copy

import cv2

from controlnet_aux import OpenposeDetector

import numpy as np

import os
import datetime
import asyncio

import DB


global hash_queue_busy
global hash_queue
hash_queue = []

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

program_start_time = time.time()

processor_busy = False

# Configuration Loading
with open('config.json', 'r') as f:
    config = json.load(f)

# Model Loading
furry_model_path = config["furry_model_path"]
sonic_model_path = config["sonic_model_path"]
aing_model_path = config["aing_model_path"]
flat2DAnimerge_model_path = config["flat2DAnimerge_model_path"]
realisticVision_model_path = config["realisticVision_model_path"]
fluffysonic_model_path = config["fluffysonic_model_path"]
furryblend_model_path = config["furryblend_model_path"]
ponydiffusion_model_path = config["ponydiffusion_model_path"]
anything_model_path = config["anything_model_path"]
sonicsdxl_model_path = config["sonicsdxl_model_path"]
everclear_model_path = config["everclear_model_path"]
abyssorangemix_model_path = config["abyssorangemix_model_path"]
# xl_pony_model_path = config["xl_pony_model_path"]

global lora_metadata_list
lora_metadata_list = []
try:
    
    upscale_model = {
        '4x': {'loaded':None, 'model_path': 'stabilityai/stable-diffusion-x4-upscaler'}
    }
    
    inpainting_models = {
        'inpainting': {'loaded':None, 'model_path': './models/inpainting/SonicDiffusionV4-inpainting.inpainting.safetensors'},
        'sonic': {'loaded':None, 'model_path': sonic_model_path},
        # 'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path},
        # 'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path},
        'sdxl-sonic': {'loaded':None, 'model_path': sonicsdxl_model_path},
        # 'anything': {'loaded':None, 'model_path': anything_model_path},
        # 'sdxl-everclear': {'loaded':None, 'model_path': everclear_model_path},
        # 'abyssorangemix': {'loaded':None, 'model_path': abyssorangemix_model_path},
    }

    txt2img_models = {
        'sonic': {'loaded':None, 'model_path': sonic_model_path},
        # 'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path},
        # 'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path},
        'sdxl-sonic': {'loaded':None, 'model_path': sonicsdxl_model_path},
        # 'anything': {'loaded':None, 'model_path': anything_model_path},
        # 'sdxl-everclear': {'loaded':None, 'model_path': everclear_model_path},
        # 'abyssorangemix': {'loaded':None, 'model_path': abyssorangemix_model_path},
    }
    
    img2img_models = {
        'sonic': {'loaded':None, 'model_path': sonic_model_path},
        # 'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path},
        # 'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path},
        'sdxl-sonic': {'loaded':None, 'model_path': sonicsdxl_model_path},
        # 'anything': {'loaded':None, 'model_path': anything_model_path},
        # 'sdxl-everclear': {'loaded':None, 'model_path': everclear_model_path},
        # 'abyssorangemix': {'loaded':None, 'model_path': abyssorangemix_model_path},
    }
    
    # txt2video_models = {
    #     'sonic': {'loaded':None, 'model_path': sonic_model_path},
    #     'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path},
    #     'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path},
    #     'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path},
    #     'furryblend': {'loaded':None, 'model_path': furryblend_model_path},
    #     'sdxl-ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path},
    #     'anything': {'loaded':None, 'model_path': anything_model_path},
    # }
    
    openpose_models = {
        'sonic': {'loaded':None, 'model_path': sonic_model_path},
        # 'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path},
        # 'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path},
        'sdxl-sonic': {'loaded':None, 'model_path': sonicsdxl_model_path},
        # 'anything': {'loaded':None, 'model_path': anything_model_path},
        # 'sdxl-everclear': {'loaded':None, 'model_path': everclear_model_path},
        # 'abyssorangemix': {'loaded':None, 'model_path': abyssorangemix_model_path},
    }
    
    
        
    # for each model in txt2img_models that doesnt have a save_pretrained folder, create one by using StableDiffusionPipeline, loading the model and using the name as the final folder:
    for model_name, model_info in txt2img_models.items():
        if not os.path.exists('./models/' + model_name):
            print("Creating folder for " + model_name)
            try:
                if model_name.startswith("sdxl-"):
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_info['model_path'],
                        torch_dtype=torch.float16,
                        revision="fp16",
                        feature_extractor=None,
                        requires_safety_checker=False,
                        cache_dir="",
                        load_safety_checker=False,
                    )
                    pipeline.save_pretrained('./models/' + model_name)
                else:
                    pipeline = StableDiffusionPipeline.from_single_file(
                        model_info['model_path'],
                        torch_dtype=torch.float16,
                        revision="fp16",
                        safety_checker=None,
                        feature_extractor=None,
                        requires_safety_checker=False,
                        cache_dir="",
                        load_safety_checker=False,
                    )
                    pipeline.save_pretrained('./models/' + model_name)
            except Exception as e:
                print(f"Failed to load the model: {e}")
                raise
                
    torch.backends.cuda.matmul.allow_tf32 = True

except Exception as e:
    print(f"Failed to load the model: {e}")
    raise























def load_embeddings(pipeline, name):
    if not name.startswith("sdxl-"):
        # pipeline.load_textual_inversion("embeddings/EasyNegativeV2.safetensors")
        # pipeline.load_textual_inversion("embeddings/BadDream.pt")
        pipeline.load_textual_inversion("embeddings/boring_e621_v4.pt")
        # pipeline.load_textual_inversion("embeddings/fcNeg-neg.pt")
        pipeline.load_textual_inversion("embeddings/fluffynegative.pt")
        pipeline.load_textual_inversion("embeddings/badyiffymix41.safetensors")
        # pipeline.load_textual_inversion("embeddings/badhandv4.pt")
        pipeline.load_textual_inversion("embeddings/gnarlysick-neg.pt")
        pipeline.load_textual_inversion("embeddings/negative_hand-neg.pt")
    # else:
        # pipeline.load_textual_inversion("embeddings/sdxl/zPDXL.pt")
        # pipeline.load_textual_inversion("embeddings/sdxl/zPDXL-neg.pt")
    return pipeline

from DeepCache import DeepCacheSDHelper

def enable_deep_cache(pipeline):
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(
        cache_interval=2,
        cache_branch_id=0,
    )
    helper.enable()

def create_and_load_inpainting_model(model_path, name, model_type, data):
    
    if name == "inpainting":
        # print("\nLoading Inpainting model")
        pipeline = StableDiffusionInpaintPipeline.from_single_file(
            pretrained_model_link_or_path="./models/inpainting/SonicDiffusionV4-inpainting.inpainting.safetensors",
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            cache_dir="",
            load_safety_checker=False,
        )
    elif name.startswith("sdxl-"):
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            './models/' + name, 
            torch_dtype=torch.float16,
            revision="fp16",
        )
    else:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            './models/' + name, 
            torch_dtype=torch.float16, 
            revision="fp16",
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            cache_dir="",
            load_safety_checker=False,
        )
        
    if data['scheduler'] == "eulera":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if data['scheduler'] == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True
        
    pipeline = load_embeddings(pipeline, name)    
    
    pipeline.to("cpu")
    
    return pipeline







def create_and_load_model(model_path, name, model_type, data):

    if name.startswith("sdxl-"):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            './models/' + name,
            # custom_pipeline="lpw_stable_diffusion_xl",
            torch_dtype=torch.float16,
            revision="fp16",
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            './models/' + name, 
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16, 
            revision="fp16",
            safety_checker=None
        )
        
    pipeline = load_embeddings(pipeline, name)
    
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    
    # if name.startswith("sdxl-"):
    #     sdxl_vae = AutoencoderKL.from_single_file(
    #         './vaes/sdxl_vae.safetensors', 
    #         torch_dtype=torch.float16,
    #         revision="fp16",
    #     )
    #     pipeline.vae = sdxl_vae
    
    
    # DOESNT WORK: 
    # if not name.startswith("sdxl-"):
    #     img2img_models[name]['loaded'] = StableDiffusionImg2ImgPipeline(**pipeline.components)
    #     inpainting_models[name]['loaded'] = StableDiffusionInpaintPipeline(**pipeline.components)
        

    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    
    enable_deep_cache(pipeline)
    
    pipeline.to("cpu")

    return pipeline

def create_and_load_img2img_model(model_path, name, model_type, data):

    if name.startswith("sdxl-"):
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            './models/' + name,
            # custom_pipeline="lpw_stable_diffusion_xl",
            torch_dtype=torch.float16,
            revision="fp16",
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            './models/' + name, 
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16, 
            revision="fp16",
            safety_checker=None
        )
    
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    pipeline = load_embeddings(pipeline, name)
    

    pipeline.to("cpu")
    
    return pipeline






# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")

# def create_and_load_txt2video_model(model_path, name, model_type, data):
    
#     pipeline = AnimateDiffPipeline.from_pretrained(
#         './models/' + name,
#         motion_adapter=adapter,
#         torch_dtype=torch.float16,
#     )
    
#     if data['scheduler'] == "eulera":
#         pipeline.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
#             './models/' + name,
#             subfolder="scheduler",
#             clip_sample=False,
#             timestep_spacing="linspace",
#             beta_schedule="linear",
#             steps_offset=1,
#         )
#     if data['scheduler'] == "dpm":
#         pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
#             './models/' + name,
#             subfolder="scheduler",
#             clip_sample=False,
#             timestep_spacing="linspace",
#             beta_schedule="linear",
#             steps_offset=1,
#         )
    
#     pipeline.unet.set_attn_processor(AttnProcessor2_0())
#     pipeline = load_embeddings(pipeline, name)

#     pipeline.enable_vae_slicing()
#     pipeline.enable_model_cpu_offload(gpu_id=0)

#     return pipeline


def create_and_load_controlnet_model(model_path, name, model_type, data):
    
    if(model_type == "openpose"):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16, use_safetensors=True)
    elif(model_type == "controlnet_img2img"):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        './models/' + name, 
        controlnet=controlnet,
        torch_dtype=torch.float16, 
        revision="fp16",
        safety_checker=None
    )
    
    pipeline.enable_vae_slicing()
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    pipeline = load_embeddings(pipeline, name)
    
    pipeline.enable_vae_tiling()
    
    components = pipeline.components
    components['safety_checker'] = None    
    
    
    pipeline.to("cpu")

    return pipeline
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
  

    
    
def get_txt2img_model(name, data):
    model_info = txt2img_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_model(model_info['model_path'], name, data['request_type'], data)
    else:
        model_info = txt2img_models[name]
        
    pipeline = model_info['loaded']
        
    if data['scheduler'] == "eulera":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if data['scheduler'] == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True
                
    return pipeline



def get_img2img_model(name, data):
    model_info = img2img_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_img2img_model(model_info['model_path'], name, data['request_type'], data)
    else:
        model_info = img2img_models[name]
        
    pipeline = model_info['loaded']
        
    if data['scheduler'] == "eulera":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if data['scheduler'] == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True
                
    return model_info['loaded']



# def get_txt2video_model(name, data):
#     if data['scheduler'] == "eulera":
#         model_info = eulera_txt2video_models[name]
#     if data['scheduler'] == "dpm":
#         model_info = dpm_txt2video_models[name]
    
#     if model_info['loaded'] is None:
#         model_info['loaded'] = create_and_load_txt2video_model(model_info['model_path'], name, data['request_type'], data)
#     else:
#         if data['scheduler'] == "eulera":
#             model_info = eulera_txt2video_models[name]
#         if data['scheduler'] == "dpm":
#             model_info = dpm_txt2video_models[name]

                
#     return model_info['loaded']








def get_inpainting_model(name, data):
    if data['inpainting_original_option'] == False:
        name = "inpainting"
    model_info = inpainting_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_inpainting_model(model_info['model_path'], name, data['request_type'], data)
    else:
        model_info = inpainting_models[name]
        
    pipeline = model_info['loaded']
        
    if data['scheduler'] == "eulera":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if data['scheduler'] == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True
                   
    return model_info['loaded']








# def get_controlnet_img2img_model(name, data):
#     model_info = controlnet_img2img_models[name]
    
#     if model_info['loaded'] is None:
#         model_info['loaded'] = create_and_load_controlnet_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
#     else:
#         model_info = controlnet_img2img_models[name]
                        
#     return model_info['loaded']



def get_openpose_model(name, data):
    
    model_info = openpose_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_controlnet_model(model_info['model_path'], name, data['request_type'], data)
    else:
        model_info = openpose_models[name]
        
    pipeline = model_info['loaded']
        
    if data['scheduler'] == "eulera":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if data['scheduler'] == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True
        
    return model_info['loaded']












from pathlib import Path

def load_yaml():
    global lora_weights_map
    try:
        with yaml_file_path.open("r", encoding="utf-8") as f:
            lora_weights_content = f.read()
        lora_weights_map = yaml.safe_load(lora_weights_content)
    except Exception as e:
        print(f"Error reading the YAML file: {e}")

def update_lora_weights_map():
    global lora_weights_map
    global yaml_file_path
    yaml_file_path = Path("./lora_weights.yaml")

    # Load the YAML file initially
    load_yaml()

    # Create a thread to monitor the YAML file for changes
    file_monitor_thread = threading.Thread(target=monitor_yaml_file, daemon=True)
    file_monitor_thread.start()

def monitor_yaml_file():
    last_modified_time = yaml_file_path.stat().st_mtime
    while True:
        try:
            current_modified_time = yaml_file_path.stat().st_mtime
            if current_modified_time != last_modified_time:
                load_yaml()
                last_modified_time = current_modified_time
        except Exception as e:
            print(f"Error monitoring the YAML file: {e}")
        time.sleep(5)

# Start the thread to update the lora_weights_map
threading.Thread(target=update_lora_weights_map, daemon=True).start()

app = Flask(__name__)
CORS(app)

@app.route('/get-lora-yaml')
def get_lora_yaml():

    return jsonify(lora_weights_map)


from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

def add_watermark(image, text, data):
    
    # create a font size based on the width and height of the image:
    data_height = image.height
    data_width = image.width
    
    text_image = Image.new('RGBA', (data_width, data_height), (255, 255, 255, 0))
    
    draw = ImageDraw.Draw(text_image)
    
    font_scale = data_height + data_width
    font_size = int(font_scale / 55)
    
    
    # if data.get('upscale', True):
    #     with open('upscale-settings.yaml', 'r') as file:
    #         upscaleSettings = yaml.safe_load(file)
    #         font_size = upscaleSettings.get('font-size', 96)
    # else:
    #     font_size = 24
        
        
    try:
        font = ImageFont.truetype("Amaranth-Regular.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Load the default font in case of error

    # Use textbbox to get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the x, y position of the text
    x = image.width - 10 - text_width  # 10 pixels offset from the right edge
    y = 10  # 10 pixels offset from the top edge

    outline_color = "black"
    outline_thickness = font_size / 20  # 1/15th of the font size
    # draw the outline using the outline thickness value:
    for i in range(1, int(outline_thickness) + 1):
        draw.text((x - i, y - i), text, font=font, fill=outline_color)
        draw.text((x + i, y - i), text, font=font, fill=outline_color)
        draw.text((x - i, y + i), text, font=font, fill=outline_color)
        draw.text((x + i, y + i), text, font=font, fill=outline_color)

    # Text itself
    text_color = (255, 255, 255)
    draw.text((x, y), text, font=font, fill=text_color)
    
    # change the text image to be lower opacity:
    text_image = text_image.convert('RGBA')
    text_image_data = text_image.getdata()
    
    new_text_image_data = []
    for item in text_image_data:
        if item[3] == 0:
            new_text_image_data.append((255, 255, 255, 0))
        if item[3] != 0 and item[2] != 255:
            new_text_image_data.append((0, 0, 0, 120))
        if item[3] != 0 and item[2] == 255:
            new_text_image_data.append((255, 255, 255, 120))
            
    text_image.putdata(new_text_image_data)
    
    # merge the text image with the original image:
    image = Image.alpha_composite(image.convert('RGBA'), text_image)
    
    # convert the image back to RGB mode:
    image = image.convert('RGB')
    
    # save image as saved_image.png:
    # image.save('saved_image.png')

    return image

class QueueRequest:
    def __init__(self, request_id, data, image_data=None):
        self.request_id = request_id
        self.data = data
        self.image_data = image_data
        self.status = "queued"

request_queue = []  # Use a list instead of the queue module for more control
# Dictionary to hold results indexed by request_id
results = {}
current_position = 0

def contains_any(string, substring_list):
    """Check if 'string' contains any of the substrings in 'substring_list'."""
    return any(sub in string for sub in substring_list)


import re

# Updated pattern to include optional minus sign before the number
loraPattern = re.compile(r"(style|effect|concept|clothing|character|pose|background)-([a-zA-Z0-9]+):(-?[0-9.]+)")

def load_loras(request_id, current_model, lora_items, data):
    global lora_metadata_list
    start_time = time.time()
    lora_metadata_list = []
    
    try:
        # Parse the prompt for Lora settings and strengths
        prompt = data.get('prompt', '')
        lora_settings = {f"{match[0]}-{match[1]}": float(match[2]) for match in loraPattern.findall(prompt)}
        
        # Remove the matched patterns from the prompt
        cleaned_prompt = re.sub(loraPattern, '', prompt)
        data['prompt'] = cleaned_prompt.strip()  # Remove leading/trailing whitespace if any
        
        # clean up any random commas that might be left over:
        data['prompt'] = data['prompt'].replace(", , ", ", ")
        
    except Exception as e:
        print(f"Error parsing prompt for Lora settings: {e}")
        lora_settings = {}

    
    
    print(f"Cleaned Prompt: {data['prompt']}\n")
    
    adapter_name_list = []
    adapter_weights_list = []

    for item in lora_items:
        time.sleep(0.1)
        try:
            category, key = item.split('-', 1)
            lora_data = lora_weights_map.get(category, {}).get(item)

            if lora_data:
                strength = lora_settings.get(item, lora_data.get('strength', 1.0))
                    
                print(f"Strength for {item}: {strength}")

                if strength:  # This checks for strength != 0; it will work with negative numbers as well
                    lora_metadata = f"{lora_data['name']} - strength: {strength}"
                    lora_metadata_list.append(lora_metadata)

                    current_model.load_lora_weights(
                        lora_data['lora'], 
                        low_cpu_mem_usage=True, 
                        ignore_mismatched_sizes=True, 
                        adapter_name=item
                    )
                    adapter_name_list.append(item)
                    adapter_weights_list.append(strength)
            else:
                print(f"No data found for {item}")
        except Exception as e:
            print(f"Error processing item '{item}': {e}")
            
    print(f"Time taken to fetch loras: {time.time() - start_time:.2f} seconds")
        
    set_adapters_start_time = time.time()

    try:
        print(f"current_model.set_adapters({adapter_name_list}, adapter_weights={adapter_weights_list})")
        current_model.set_adapters(adapter_name_list, adapter_weights=adapter_weights_list)
        current_model.fuse_lora()
        return current_model
    except Exception as e:
        print(f"Error during model configuration: {e}")
        
    print(f"Time taken to set adapters: {time.time() - set_adapters_start_time:.2f} seconds")
    










def process_image(current_model, model_type, data, request_id, save_image=False):
    try:
        
        current_model.to("cuda:0")

        generator = torch.Generator(device="cuda")
        data['seed'] = generator.manual_seed(data['seedNumber'])
        
        if model_type == 'txt2img':
            with torch.inference_mode():
                outputs = current_model(
                    prompt=data['prompt'],
                    negative_prompt=data['negative_prompt'],
                    num_images_per_prompt=data['image_count'],
                    num_inference_steps=data['steps'],
                    width=data['width'],
                    height=data['height'],
                    guidance_scale=data['guidance'],
                    generator=data['seed'],
                ).images
        if model_type == 'img2img':
            with torch.inference_mode():
                outputs = current_model(
                    prompt=data['prompt'],
                    negative_prompt=data['negative_prompt'],
                    num_inference_steps=data['steps'],
                    width=data['width'],
                    height=data['height'],
                    guidance_scale=data['guidance'],
                    generator=data['seed'],
                    strength=data['strength'],
                    image=data['image_data'],
                    num_images_per_prompt=data['image_count'],
                ).images
        elif model_type == 'inpainting':
            # print("Inpainting Generation Process Started")
            with torch.inference_mode():
                outputs = current_model(
                    prompt=data['prompt'],
                    negative_prompt=data['negative_prompt'],
                    image=data['image_data'],
                    mask_image=data['mask_data'],
                    width=data['width'],
                    height=data['height'],
                    guidance_scale=data['guidance'],
                    generator=data['seed'],
                    num_images_per_prompt=data['image_count'],
                    num_inference_steps=data['steps'],
                    strength=data['strength'],
                    ).images
        elif model_type == 'controlnet_img2img' or model_type == 'openpose':
            with torch.inference_mode():
                outputs = current_model(
                    prompt=data['prompt'],
                    negative_prompt=data['negative_prompt'],
                    num_inference_steps=data['steps'],
                    width=data['width'],
                    height=data['height'],
                    guidance_scale=data['guidance'],
                    generator=data['seed'],
                    strength=data['strength'],
                    image=data['image_data'],
                    num_images_per_prompt=data['image_count'],
                ).images
        elif model_type == 'txt2video':
            with torch.inference_mode():
                outputs = current_model(
                    prompt=data['prompt'],
                    negative_prompt=data['negative_prompt'],
                    width=data['width'],
                    height=data['height'],
                    num_inference_steps=data['steps'],
                    guidance_scale=data['guidance'],
                    generator=data['seed'],
                    num_frames=data['video_length'],
                )

        return outputs
                
            

    except Exception as e:
        current_model.to("cpu")
        error_message = str(e)
        error_message = error_message.replace("'", '"')
        if error_message == '"LayerNormKernelImpl" not implemented for "Half"':
            if data['request_type'] == 'txt2img':
                txt2img_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'img2img':
                img2img_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'inpainting':
                inpainting_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'openpose':
                openpose_models[data['model']]['loaded'] = None
            error_message = error_message + " | Model Reloaded"
        
        error_message = error_message + " | Reset Model"
        print("Error processing request:", error_message)
        results[request_id] = {"status": "error", "message": error_message}
        return "CONTINUE"
    
# model_instance = get_txt2img_model('furry')
# process_image(model_instance, 'txt2img', {'prompt': '1girl, amy rose, glossy skin, shiny skin, (masterpiece, soft lighting, studio lighting, high quality, high detail, detailed background), in city, neon, glowing, rainging, bright, cute, fluffy, furry, wearing thigh highs, wearing croptop, looking at viewer, tan belly, bloom, bokeh, lens flare, sunlight, rainbow, crowded street path, street light, ', 'negative_prompt': '', 'image_count': 1, 'steps': 20, 'width': 512, 'height': 512, 'guidance': 6, 'seed': generator.manual_seed(69420) }, 'test', save_image=True)
        
        
def is_image_nearly_black(image, threshold=10):
    """
    Check if the given image is nearly black.
    Args:
    - image (PIL.Image): The image to check.
    - threshold (int): The threshold for average pixel value to consider as nearly black.
    Returns:
    - bool: True if the image is nearly black, else False 
    """
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    avg_pixel_value = np_image.mean()
    return avg_pixel_value < threshold

def add_metadata(image, metadata):
    meta = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        meta.add_text(str(key), str(value))
    return meta











def save_image(request_id, output_image, model_type, data, image_index=0, font_size=20):
    try:
        # Simplify accountId handling
        accountId_string = "" if data.get('accountId') == "0" else data.get('accountId', '')

        # Generate metadata
        metadata = {
            "request_id": request_id,
            "model_type": model_type,
            "prompt": data['true_prompt'],
            "negative_prompt": str(data['negative_prompt']),
            "loras": lora_metadata_list,  
            "steps": data['steps'],
            "seed": data['seedNumber'],
            "CFG": data['guidance'],
            "model": data['model'],
            "upscaled": data['upscale'],
            "generation_date": datetime.datetime.now().isoformat(),
            "accountId": str(data['accountId']),
            "scheduler": data['scheduler']
        }

        # Add watermark, if applicable
        if model_type != "txt2video":
            # watermarks = ["JSCammie.com", "Cammie.ai", "Check out mobians.ai!", "In femboys we trust", "NeverSFW.gg"]
            # watermarks_chances = [0.5, 0.1, 0.1, 0.1, 0.1]
            # watermark_text = random.choices(watermarks, watermarks_chances)[0]
            watermark_text = "JSCammie.com"
            watermarked_image = add_watermark(output_image, watermark_text, data)
            meta = add_metadata(watermarked_image, metadata)
            buffered = io.BytesIO()
            watermarked_image.save(buffered, format="PNG", pnginfo=meta)
            img_str = base64.b64encode(buffered.getvalue()).decode()
        else:
            img_str = data['video_string']

        return {
            "width": output_image.width,
            "height": output_image.height,
            "base64": img_str
        }
            
    except Exception as e:
        print(f"Error saving image: {e}")
        return None












log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def generate_error_response(message, status_code):
    response = jsonify({"status": "error", 'message': message})
    response.status_code = status_code
    return response

def round_to_multiple_of_eight(number):
    """Round a number to the nearest multiple of 8."""
    # print("Rounding to multiple of 8: ", number)
    return round(number / 8) * 8






def export_to_mp4(pil_images, output_filename, fps=10):
    """
    Converts a list of PIL images to an MP4 video file.

    Args:
    - pil_images: List of PIL Image objects.
    - output_filename: The name of the output MP4 video file.
    - fps: Frames per second for the output video.
    """
    # Convert PIL images to NumPy arrays
    image_arrays = [np.array(img) for img in pil_images]
    
    # Create a video clip from the image arrays
    clip = ImageSequenceClip(image_arrays, fps=fps)
    
    # Write the video clip to a file
    clip.write_videofile(output_filename, codec='libx264')
    
    # load the video file into memory and return it as a base64 string
    with open(output_filename, "rb") as file:
        video_data = file.read()
        video_base64 = base64.b64encode(video_data).decode("utf-8")
        
    return video_base64
    
    
    
    
    

def process_request(queue_item):
    # print("Processing request:", queue_item.request_id)
    try:
        start_time = time.time()
        nextReq = (" | Queue Length: " + str(len(request_queue)))
        print("\nProcessing next request..." + nextReq)
        request_id = queue_item.request_id
        data = queue_item.data
        
        promptString = str(data['prompt'])
        
        # if prompt is a list then join it into a string:
        if isinstance(data['prompt'], list):
            promptString = ' '.join(data['prompt'])
        
        # data on multiple print lines for easier debugging
        print("Request Type: " + str(data['request_type']) + " | Model: " + str(data['model']) + " Scheduler: " + str(data['scheduler']) + "\nSteps: " + str(data['steps']) + " | Width: " + str(data['width']) + "px | Height: " + str(data['height']) + "px\nSeed: " + str(data['seedNumber']) + " | Strength: " + str(data['strength']) + " | CFGuidance: " + str(data['guidance']) + " | Image Count: " + str(data['image_count'] )  + "\nPrompt: " + str(promptString) + "\nNegative Prompt: " + str(data['negative_prompt']) + "\nLora: " + str(data['lora']))

        model_name = data['model']
        lora = data.get('lora', "NO")
        model_type = data['request_type']
        
                
        # if model_type is txt2img or img2img, get the model, else get the inpainting model:
        if model_type == 'txt2img':
            model = get_txt2img_model(model_name, data)
        elif model_type == 'img2img':
            model = get_img2img_model(model_name, data)
        elif model_type == 'inpainting':
            model = get_inpainting_model(model_name, data)
        # elif model_type == 'controlnet_img2img':
        #     model = get_controlnet_img2img_model(model_name, data)
        elif model_type == 'openpose':
            model = get_openpose_model(model_name, data)
        # elif model_type == 'latent_couple':
        #     model = get_txt2img_model(model_name, data)
        #     data['inpainting_original_option'] = True
        #     inpainting_model = get_inpainting_model(model_name, data)
        #     img2img_model = get_img2img_model(model_name, data)
            
        current_model = model
            
        if current_model is not None:
            current_model.unfuse_lora()
            current_model.unload_lora_weights()
            print(model_type)
            if data['lora'] is not None and model_type != 'latent_couple':
                current_model = load_loras(request_id, current_model, lora, data)
                print("loras loaded successfully")
            if model_type != 'latent_couple':
                model_outputs = process_image(current_model, model_type, data, request_id)
                current_model.to("cpu")
            # else:
            #     inpainting_model.unfuse_lora()
            #     inpainting_model.unload_lora_weights()
            #     print("Latent Couple Generation Process Started")
                
            #     data['quantity'] = 1
                
            #     prompt_og = data['prompt']
                
            #     data['prompt'] = prompt_og[0]
                
            #     load_loras(request_id, inpainting_model, lora, data)
                
            #     base_images = process_image(current_model, 'txt2img', data, request_id)  
                
            #     current_model.to("cpu")
                
            #     for i, base_image in enumerate(base_images):
            #         base_image.save(f"base{i}.png")
            
            #     if data['steps'] > 50:
            #         data['steps'] = 50
                               
            #     slices = []
                
            #     base_image = base_images[0]

            #     for i in range(data['splits']):
            #         if data['splitType'] == "horizontal":
            #             slice_width = base_image.width / data['splits']
            #             overlap_width = slice_width * data['splitOverlap']
            #             slice_width = round_to_multiple_of_eight(slice_width + overlap_width)

            #             left = i * (base_image.width / data['splits']) - (overlap_width if i > 0 else 0)
            #             right = left + slice_width
            #             sliceImageMask = Image.new('RGB', (int(base_image.width), int(base_image.height)), (0, 0, 0))
            #             round_to_multiple_of_eight(left)
            #             round_to_multiple_of_eight(right)
            #             sliceImageMask.paste((255, 255, 255), (int(left), 0, int(right), int(base_image.height)))
            #             slices.append(sliceImageMask)
            #         else:  # for vertical split
            #             slice_height = base_image.height / data['splits']
            #             overlap_height = slice_height * data['splitOverlap']
            #             slice_height = round_to_multiple_of_eight(slice_height + overlap_height)

            #             top = i * (base_image.height / data['splits']) - (overlap_height if i > 0 else 0)
            #             bottom = top + slice_height
                        
            #             sliceImageMask = Image.new('RGB', (int(base_image.width), int(base_image.height)), (0, 0, 0))
            #             round_to_multiple_of_eight(top)
            #             round_to_multiple_of_eight(bottom)
            #             sliceImageMask.paste((255, 255, 255), (0, int(top), int(base_image.width), int(bottom)))
            #             slices.append(sliceImageMask)
                                                        
            #     model_outputs = []
                        
            #     # Save all the mask slices with random names and process each base image with each mask
            #     for i, mask_slice in enumerate(slices):
            #         mask_slice.save(f"slice{i}.png")
            #         data['mask_data'] = mask_slice
                    
            #         data['prompt'] = prompt_og[i + 1]
            #         print("Prompt: ", data['prompt'])
                    
            #         inpainting_model.unfuse_lora()
            #         inpainting_model.unload_lora_weights()
                    
            #         load_loras(request_id, inpainting_model, lora, data)

            #         processed_images_for_this_mask = []
            #         for j, base_image in enumerate(base_images):
            #             data['image_data'] = base_image
            #             output = process_image(inpainting_model, 'inpainting', data, request_id)
                        
            #             output[0].save(f"processed_slice{j}-{i}.png")

            #             processed_image = output[0] if isinstance(output, list) else output
            #             processed_images_for_this_mask.append(processed_image)
                        
            #         inpainting_model.to("cpu")

            #         # Update base_images for the next iteration of mask slices
            #         base_images = processed_images_for_this_mask
                    
            #     data['image_data'] = base_images[0]
                
            #     promptString = ""
                
            #     for prompt in prompt_og:
            #         promptString += prompt + " "

            #     with open('latent-loopback-settings.yaml', 'r') as file:
            #         loopback_data = yaml.safe_load(file)
                    
            #     if loopback_data['prompt_override'] is not None:
            #         data['prompt'] = prompt_og[0]
            #     else:
            #         data['prompt'] = promptString

            #     # Access the data
            #     data['steps'] = loopback_data['steps']
            #     data['strength'] = loopback_data['strength']
                
            #     print("BEFORE IMG2IMG PASS")
                
            #     if loopback_data['enabled'] is True:
            #         img2img_model.unfuse_lora()
            #         img2img_model.unload_lora_weights()
            #         print("Unfused and unloaded loras for img2img model")
            #         load_loras(request_id, img2img_model, lora, data)
            #         print("Loras loaded for img2img model")

            #         model_outputs = process_image(img2img_model, 'img2img', data, request_id)
            #         img2img_model.to("cpu")
            #     # get the type of model_outputs:
                
            #     promptString = ""
                
            #     for prompt in prompt_og:
            #         promptString += prompt + " "

            #     data['prompt'] = promptString       
            #     data['image_data'] = None      

            if model_outputs == "CONTINUE":
                error_message = "INPAINTING FAILED"
                # convert ' to " in the error_message: 
                print("Error processing request:", error_message)
                results[request_id] = {"status": "error", "message": error_message}
                queue_item.status = "error"
                return "skipped"
            print("\n")
            if data['lora'] is not None:
                current_model.unfuse_lora()
                current_model.unload_lora_weights()
            # if model_type == 'latent_couple':
            #     if data['lora'] is not None:
            #         inpainting_model.unfuse_lora()
            #         inpainting_model.unload_lora_weights()
            #         if loopback_data['enabled'] is True:
            #             img2img_model.unfuse_lora()
            #             img2img_model.unload_lora_weights()
                        
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if model_outputs is not None:
                image_data_list = []
                
                if model_type == 'txt2video':
                    model_outputs = model_outputs.frames[0]
                    
                if model_type == 'txt2video':
                    
                    img2img_model_video = get_img2img_model(model_name, data)
                    
                    steps_before = data['steps']
                    strength_before = data['strength']
                    
                    data['steps'] = 30
                    data['strength'] = 0.1
                    
                    improved_frames = []
                    
                    for img in model_outputs:
                        data['image_data'] = img
                        output = process_image(img2img_model_video, "img2img", data, request_id)
                        improved_frames.append(output[0])
                        
                    img2img_model_video.to("cpu")
                        
                    model_outputs = improved_frames
                        
                        
                    data['steps'] = steps_before
                    data['strength'] = strength_before
            
            
                
                if data.get('upscale', False):
                    
                    # make the directories if they don't exist:
                    if not os.path.exists("toupscale"):
                        os.makedirs("toupscale")
                        
                    if not os.path.exists("upscaled"):
                        os.makedirs("upscaled")
                        
                    # remove the toupscale images:
                    for files in os.listdir("toupscale"):
                        os.remove(f"toupscale/{files}")
            
                    # save image as "og_image.png"
                    for index, img in enumerate(model_outputs):
                        img.save(f"toupscale/og-image-{index}.png")
                
                    import subprocess

                    # Define the command to run as a list of arguments
                    command = ["./esrganvulkan/realesrgan-ncnn-vulkan.exe", "-n", "realesrgan-x4plus-anime", "-i", "toupscale", "-o", "upscaled", "-f", "png", "-s", "4", "-t", "256"]
                    # python Real-ESRGAN/inference_realesrgan.py -n realesrgan-x4plus-anime -i og_image.png -o upscaled_image.png -f png

                    # Run the command and wait for it to complete, capturing the output
                    result = subprocess.run(command, capture_output=True, text=True)

                    # create array with the upscaled images:
                    upscaled_images = []
                    
                    # load the upscaled images into memory:
                    for index, img in enumerate(model_outputs):
                        img = Image.open(f"upscaled/og-image-{index}.png")
                        upscaled_images.append(img)
                        
                    model_outputs = upscaled_images
                    
                    
                # if model_type == 'txt2video':
                #     processed_frames = []
                #     for frame in model_outputs:
                #         frame = add_watermark(frame, "JSCammie.com", data)
                #         processed_frames.append(frame)
                        
                #     data['video_string'] = export_to_mp4(processed_frames, "animation.mp4")
                    
                #     model_outputs = [model_outputs[0]]
                
                timeBeforeSave = time.time()
                
                PIL_Images = []
                                    
                for index, img in enumerate(model_outputs):
                    image_data = save_image(request_id, img, model_type, data, index)
                    image_data_list.append(image_data)

                    if model_type != "txt2video":
                        img_bytes = base64.b64decode(image_data['base64'])
                        PIL_Images.append(Image.open(io.BytesIO(img_bytes)))

                hash_object = {
                    "data": data,
                    "images": PIL_Images,
                    "status": "pending"
                }
                hash_queue.append(hash_object)
                    
                    
                print("Time to save images: " + str(time.time() - timeBeforeSave))

            results[request_id] = {
                "images": image_data_list,
                "additionalInfo": {
                    "seed": data['seedNumber'],
                    "executiontime": time.time() - start_time,
                    "loras": lora_metadata_list
                }
            }
            
            queue_item.status = "completed"
            print_queue()
            return "processed"
            
        else:
            error_message = "Invalid model name"
            print("Error processing request:", error_message)
            results[request_id] = {"status": "error", "message": error_message}

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        results[request_id] = {"status": "error", "message": error_message}

def print_queue():
    print("\nPrinting queue...")
    print("Queue Length: " + str(len(request_queue)))
        
# def process_requests():
#     with ThreadPoolExecutor(max_workers=2) as executor:  # DESPISE THIS, FUCK MAN
#         while True:
            
#             time.sleep(0.1)  # Short sleep to prevent CPU overutilization
#             for queue_item in request_queue:
#                 if queue_item.status == "queued":
#                     # Safely get 'attempts' value, default to 0 if not found
#                     attempts = queue_item.data.get('attempts', 0)
#                     if attempts > 5:
#                         print("Removing errored request:", queue_item.request_id)
#                         request_queue.remove(queue_item)
#                         print_queue()
#                         break

#                     # Increment attempts and update status
#                     queue_item.data['attempts'] = attempts + 1
#                     queue_item.status = "waiting"
#                     queue_item.data['submitted'] = time.time()
                    
#                     executor.submit(process_request, queue_item)

#                 elif queue_item.status in ["completed", "error", "skipped"]:
#                     print(f"Removing {queue_item.status} request:", queue_item.request_id)
#                     request_queue.remove(queue_item)
#                     print_queue()
#                     break

#             # Sleep if no unprocessed request is found
#             if not any(item.status == "queued" for item in request_queue):
#                 time.sleep(0.1)

# # Start the process_requests thread
# threading.Thread(target=process_requests, daemon=True).start()

import time

hash_queue_busy = False

async def process_hash_queue():
    global hash_queue_busy
    while True:
        await asyncio.sleep(0.1)  # Corrected to use await
        if not hash_queue_busy:
            if hash_queue:  # Check if the queue is not empty
                queue_item = hash_queue[0]  # Get the first item
                if queue_item['status'] == "pending":
                    hash_queue_busy = True
                    # Assuming DB.process_images_and_store_hashes is an async function
                    await DB.process_images_and_store_hashes(queue_item['images'], queue_item['data'])
                    print("Finished processing")
                    hash_queue_busy = False
                    hash_queue.remove(queue_item)

        # Sleep if no unprocessed request is found
        if not any(item['status'] == "pending" for item in hash_queue):
            await asyncio.sleep(0.5)  # Corrected to use await

def process_queue():
    global processor_busy
    
    while True:
        time.sleep(0.1)  # Short sleep to prevent CPU overutilization
        if not processor_busy:
            if request_queue:  # Check if the queue is not empty
                queue_item = request_queue[0]  # Get the first item
                if queue_item.status == "queued":
                    queue_item.status = "waiting"
                    processor_busy = True
                    result = process_request(queue_item)
                    if result == "processed":
                        processor_busy = False
                        request_queue.remove(queue_item)  # Remove the item from the queue
                    elif queue_item.status in ["completed", "error", "skipped"]:
                        processor_busy = False
                        request_queue.remove(queue_item)  # Remove the item from the queue

        # Sleep if no unprocessed request is found
        if not any(item.status == "queued" for item in request_queue):
            time.sleep(0.5)

# Start the process_queue thread
threading.Thread(target=process_queue, daemon=True).start()


















def update_fast_passes_map():
    global fast_passes_map
    while True:
        try:
            with open('./fast_passes.yaml', 'r', encoding='utf-8') as f:
                fast_passes_content = f.read()
            # print("Lora Weights Content:", lora_weights_content)  # Debugging line
            fast_passes_map = yaml.safe_load(fast_passes_content)

        except Exception as e:
            print(f"Error reading the fast pass yaml file: {e}")
        time.sleep(1)
        
# Start a separate thread that updates the lora_weights_map
threading.Thread(target=update_fast_passes_map, daemon=True).start()

def check_fast_pass(fastpass, validated_data):
    if fastpass is None:
        return False, None

    # Remove all numbers from the fastpass string
    fastpass = ''.join([i for i in fastpass if not i.isdigit()])

    if str(fastpass) == "":
        print("Fast pass is nonexistent or the same as the accountId.")
        return False, None
    
    

    # Ensure thread-safe read with threading.Lock()
    with threading.Lock():
        fastpass_data = fast_passes_map.get('passes', {}).get(fastpass, {})
        

    # Check if the fastpass is enabled
    if fastpass_data.get('enabled', False) == False:
        return False, "Fastpass is invalid"

    # Check if the discordId matches
    if int(fastpass_data.get('discordId', 0)) != int(validated_data['accountId']):
        return False, "Fastpass is invalid"

    # Check if the fast pass has not expired
    if time.time() > int(fastpass_data.get('expires', 0)):
        return False, "Fastpass is invalid"

    print(f"Fast pass {fastpass} is valid and active.")
    return True, None
    
    
    
    
def randomize_string(input_string):
    
    if input_string.count("{") == 0:
        return input_string
    
    if input_string.count("}") == 0:
        return input_string
    
    # Check for matching number of opening and closing brackets
    if input_string.count("{") != input_string.count("}"):
        return {"status": "error", "message": "Mismatched brackets"}

    segments = input_string.split("{")
    new_string = segments[0]  # Start with the first segment

    for segment in segments[1:]:
        if "}" not in segment:
            return {"status": "error", "message": "Mismatched brackets"}

        options, rest = segment.split("}", 1)
        choice = random.choice(options.split("|"))
        new_string += choice + rest

    return new_string
    
    
    
    
def update_settings():
    global global_settings
    while True:
        try:
            with open('./global_settings.yaml', 'r', encoding='utf-8') as f:
                settings_content = f.read()
            # print("Lora Weights Content:", lora_weights_content)  # Debugging line
            global_settings = yaml.safe_load(settings_content)

        except Exception as e:
            print(f"Error reading the settings yaml file: {e}")
        time.sleep(1)
    
# Start a separate thread that updates the settings:
threading.Thread(target=update_settings, daemon=True).start()
    
def update_banned_map():
    global banned_users_map
    while True:
        try:
            with open('./banned_users.yaml', 'r', encoding='utf-8') as f:
                banned_users_content = f.read()
            # print("Lora Weights Content:", lora_weights_content)  # Debugging line
            banned_users_map = yaml.safe_load(banned_users_content)

        except Exception as e:
            print(f"Error reading the fast pass yaml file: {e}")
        time.sleep(1)
        
# Start a separate thread that update_banned_map
threading.Thread(target=update_banned_map, daemon=True).start()

def check_banned_users(userid, request_type):
    print(f"Checking fast pass: {userid} against the fast pass map: {fast_passes_map}")
    
    # Ensure thread-safe read
    with threading.Lock():
        # Check if the fastpass is in the 'passes' dictionary and if 'enabled' is True
        # return fast_passes_map.get('passes', {}).get(userid, {}).get('enabled', False)
        
        # check if the userid is in the banned_users_map, there are 4 categories of banned users:
        # txt2img, img2img, inpainting, txt2video:
        
        # txt2img: {
        #     ""
        # }

        # img2img: {
        #     ""
        # }

        # txt2video: {
        #     124950405745475588: { reason: "CP" }
        # }

        # inpainting: {
        #     ""
        # }
        
        if banned_users_map[request_type].get(userid, None) is not None:
            return banned_users_map[request_type][userid]
        else:
            return False

def load_blockedwords():
    global blockedwords
    while True:
        try:
            with open('./blockedwords.yaml', 'r', encoding='utf-8') as f:
                banned_words_content = f.read()
            # print("Lora Weights Content:", lora_weights_content)  # Debugging line
            blockedwords = yaml.safe_load(banned_words_content)

        except Exception as e:
            print(f"Error reading the banned words yaml file: {e}")
        time.sleep(1)
        
# Start a separate thread that updates the blockedwords:
threading.Thread(target=load_blockedwords, daemon=True).start()
        



def validate_input_data(data):
    validate_start_time = time.time()
    true_prompt = data['prompt']
    data['prompt'] = data['prompt'].replace('\r', '').replace('\n', '')

    # remove all numbers and () brackets from the prompt string
    filter_prompt = ''.join([i for i in data['prompt'] if not i.isdigit()])
    filter_prompt = filter_prompt.replace("(", "").replace(")", "")

    # Split the filter_prompt into a list of words, considering common separators
    words_in_prompt = set(filter_prompt.lower().split())  # Convert to lowercase for case-insensitive match

    # Check if the set of words in filter_prompt contains any of the blocked words exactly
    # The intersection will be non-empty if there are common elements
    sus_word = bool(words_in_prompt.intersection(set(blockedwords['blocked-nsfw'])))
    nsfw_word = bool(words_in_prompt.intersection(set(blockedwords['nsfw-words'])))
    
    data['strength'] = float(data.get("strength", 0.85))

    if sus_word and nsfw_word:
        return None, "Your prompt contains words that are not allowed, please remove them and try again."

    data['accountId'] = data.get('accountId', 0)

    if data['accountId'] == "":
        data['accountId'] = 0

    data['accountId'] = int(data['accountId'])

    data['prompt'] = randomize_string(data['prompt'])

    if str(data['prompt']) == "{'status': 'error', 'message': 'Mismatched brackets'}":
        return None, "Mismatched brackets ('{}' brackets are used to denote a random choice, and must be used in pairs, here is an example of a correct usage: '{woman|man} with {long|short} hair')"

    # if int(data['width']) > 512 and int(data['height']) > 512:
    #     if data['request_type'] == "txt2video":
    #         data['width'] = 512
    #         data['height'] = 512

    # if data['request_type'] == 'latent_couple':
    #     if 'AND' in data['prompt']:
    #         data['prompt'] = [s.strip() for s in data['prompt'].split('AND') if s.strip()]
    #     else:
    #         return None, "When using latent couple you need to use the keyword 'AND' to separate the different things you want in the scene"

    #     splits = data.get('splits', None)
    #     if splits is not None:
    #         splits = int(splits)
    #         data['splits'] = splits
    #         if splits < 1 or splits > 8:
    #             return None, "Splits needs to be between 1 and 8"
    #     else:
    #         return None, "You need to specify a number of splits"

    #     split_type = data.get('splitType', None)
    #     if split_type is not None:
    #         if split_type != "horizontal" and split_type != "vertical":
    #             return None, "Split direction needs to be either horizontal or vertical"
    #     else:
    #         return None, "You need to specify a split direction"

    #     if len(data['prompt']) != splits + 1:
    #         return None, "You need to specify a prompt for each split by using the 'AND' keyword"

    #     if float(data['strength']) != 1:
    #         steps_after_strength_apply = data['steps'] - (int(data['steps']) * float(data['strength']))
    #         if steps_after_strength_apply < 1:
    #             while steps_after_strength_apply < 1:
    #                 data['steps'] += 1
    #                 steps_after_strength_apply = data['steps'] - (int(data['steps']) * float(data['strength']))
    #             data['steps'] = data['steps'] * 2
    #             data['steps'] += 1
    #             data['steps'] = round(data['steps'])

    # if not data['model'].startswith("sdxl-"):
    #     if data['height'] > 1024 or data['width'] > 1024:
    #         return None, "Image dimensions are too large. Please use an image with a maximum resolution of 1024x1024."

    if data['steps'] > 126:
        return None, "You have reached the limit of 125 steps per request. Please reduce the number of steps and try again."

    if data['quantity'] > 5:
        return None, "You have reached the limit of 4 images per request. Please reduce the number of images and try again."

    if data['steps'] > 20 and data['quantity'] > 5:
        return None, "You have reached the limit of 4 images per request. Please reduce the number of steps and try again."

    if len(data['lora']) > 6:
        return None, "You have reached the limit of 5 Lora options. Please deselect some and try again."

    if data['seed'] is None:
        data['seed'] = -1

    if int(data['seed']) > 2**32 - 1:
        return None, "Seed is too large. Please use a seed between -1 and 4294967295."

    if int(data['seed']) < -1:
        return None, "Seed is too small. Please use a seed between -1 and 4294967295."

    if data['request_type'] == 'txt2video':
        if int(data['video_length']) > 16 or int(data['video_length']) < 6:
            return None, "Video length is too long/short. Please use a video length between 6 and 16 frames."
        if int(data['steps']) > 50:
            return None, "text 2 video is limited to 50 steps! Please reduce the number of steps and try again."


    if data.get("model", "sonic").startswith("sdxl-"):
        
        negative_embedding_words_sdxl = ""
        
        # negative_embedding_words_sdxl = "zPDXL-neg, "
        # positive_embedding_words_sdxl = "zPDXL, "
        # data['prompt'] = positive_embedding_words_sdxl + data.get("prompt", "")
        negative_prompt_final = negative_embedding_words_sdxl + data.get("negativeprompt", "")
        
    else:
        
        negative_embedding_words_sd15 = "boring_e621_v4, fcNeg, fluffynegative, badyiffymix41, gnarlysick-neg, negative_hand-neg, "
        negative_prompt_final = negative_embedding_words_sd15 + data.get("negativeprompt", "")
                
    data['negative_prompt'] = negative_prompt_final
        
    if data['strength'] > 1:
        data['strength'] = data['strength'] / 100
    
    if data['strength'] < 0:
        data['strength'] = 0.01
        
    data['image'] = data.get("image", None)
    data['inpaintingMask'] = data.get("inpaintingMask", None)

    if data['image'] is not None:
        try:
            base64_encoded_data = data['image'].split(',', 1)[1]
            image_data = base64.b64decode(base64_encoded_data)
            img_bytes = io.BytesIO(image_data)
            data['image'] = Image.open(img_bytes)
            
            print("Image width height before")
            
            # Determine the scaling factor to ensure both sides are at least 512px
            scale_factor = max(512 / data['image'].width, 512 / data['image'].height)
            
            print("Image width height after")

            # Calculate new dimensions
            new_width = round_to_multiple_of_eight(data['image'].width * scale_factor)
            new_height = round_to_multiple_of_eight(data['image'].height * scale_factor)


            # Update dimensions in the data dictionary
            data['width'], data['height'] = new_width, new_height

            # Resize the image
            data['image'] = data['image'].resize((new_width, new_height))
            data['image'] = data['image'].convert('RGB')


        except Exception as e:
            return generate_error_response("Failed to identify image file", 400)
    else:
        data['image'] = None
        
        
        
    if data['inpaintingMask'] is not None:
        try:
            base64_encoded_data = data['inpaintingMask'].split(',', 1)[1]
            mask_data = base64.b64decode(base64_encoded_data)
            img_bytes = io.BytesIO(mask_data)
            data['inpaintingMask'] = Image.open(img_bytes)
            
            data['inpaintingMask'] = data['inpaintingMask'].resize((round(data['width']), round(data['height'])))
            data['inpaintingMask'].save("inpaintingBefore.png")
        except Exception as e:
            return generate_error_response("Failed to identify image file", 400)
    else:
        data['inpaintingMask'] = None

    if int(data['seed']) == -1:
        data['seedNumber'] = random.randint(0, 2**32 - 1)
    else:
        data['seedNumber'] = int(data['seed'])
        
    data['seed'] = data['seedNumber']

    validated_data = {
        'model': data.get('model'),
        'prompt': data.get('prompt'),
        'negative_prompt': negative_prompt_final,
        'image_count': int(data.get("quantity")),
        'steps': int(data.get("steps", 20)),
        'width': int(data.get("width", 512)),
        'height': int(data.get("height", 512)),
        'seed': int(data.get("seed", -1)),
        'strength': float(data.get("strength", 0.75)),
        'guidance': float(data.get("guidance", 5)),
        'image_data': data.get("image", None),
        'mask_data': data.get("inpaintingMask", None),
        'lora': data.get('lora', None),
        'lora_strengths': data.get('lora_strengths', None),
        'enhance_prompt': data.get('enhance_prompt', False),
        'request_type': data['request_type'],
        'upscale': data.get('upscale', False),
        'inpainting_original_option': data.get('inpainting_original_option', False),
        'splitType': data.get('splitType', "horizontal"),
        'splits': int(data.get('splits', 1)),
        'splitOverlap': float(data.get('splitOverlap', 0.1)),
        'finalStrength': float(data.get('finalStrength', 0.2)),
        'video_length': int(data.get('video_length', 16)),
        'accountId': int(data.get('accountId', 0)),
        'true_prompt': str(true_prompt),
        'scheduler': data.get('scheduler', "eulera"),
        'fastpass': data.get('fastpass', None),
        'seedNumber': int(data['seedNumber']),
    }
    
    if validated_data['request_type'] == "inpainting":
        validated_data['strength'] = 0.75
        

    return validated_data, None




def prepare_request_data(validated_data):
    return validated_data

@app.route('/cancel_request/<request_id>', methods=['GET'])
def cancel_request(request_id):
    try:
        for item in request_queue:
            if item.request_id == request_id:
                if item.status == "waiting" or item.status == "processed":
                    return jsonify({"status": "processing", "message": "Request is currently being processed"}), 200
                item.status = "cancelled"
                request_queue.remove(item)
                return jsonify({"status": "cancelled", "message": "Cancelled Request"}), 200
        return jsonify({"status": "not found", "message": "Invalid request_id"}), 404
    except Exception as e:
        return generate_error_response(str(e), 500)

@app.route('/queue_position/<request_id>', methods=['GET'])
def check_queue_position(request_id):
    # Loop through the queue and find the position for the given request_id
    for index, item in enumerate(request_queue):
        if item.request_id == request_id:
            
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": len(request_queue)}), 200
    if request_id in results:
        if results[request_id].get("status") == "error":
            return jsonify({"status": "error", "message": results[request_id].get("message")}), 200
        return jsonify({"status": "completed", "request_id": request_id}), 200
    return jsonify({"status": "not found", "message": "Invalid request_id"}), 404

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    result = results.get(request_id)
    if result:
        return jsonify(result)
    else:
        return jsonify({"status": "processing"}), 202 

                
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        if global_settings.get('maintenance', False):
            return generate_error_response("Maintenance Mode is currently enabled, the requests that are already in the queue are being completed, please wait a minute or two and try again.", 503)

        data = request.json
        
        if global_settings.get('sdxl', False):
            if data['model'].startswith("sdxl-"):
                return generate_error_response("SDXL is currently disabled, please use the other models instead.", 503)
            
        # cap the steps to global_settings['sdxl_max_steps']:
        if data['model'].startswith("sdxl-"):
            if data['steps'] > global_settings['sdxl_max_steps']:
                data['steps'] = global_settings['sdxl_max_steps']
                
        if data.get('aspect_ratio', None) is not None:
            if data['aspect_ratio'] == "portrait":
                data['width'] = 512
                data['height'] = 768
                
            elif data['aspect_ratio'] == "landscape":
                data['width'] = 768
                data['height'] = 512
                
            elif data['aspect_ratio'] == "square":
                data['width'] = 512
                data['height'] = 512
                
            elif data['aspect_ratio'] == "square++":
                data['width'] = 768
                data['height'] = 768
                
            elif data['aspect_ratio'] == "bannerHorizontal":
                data['width'] = 1024
                data['height'] = 512
                
            elif data['aspect_ratio'] == "bannerVertical":
                data['width'] = 512
                data['height'] = 1024
                
            elif data['aspect_ratio'] == "16:9":
                data['width'] = 1024
                data['height'] = 576
                
            elif data['aspect_ratio'] == "9:16":
                data['width'] = 576
                data['height'] = 1024
                
            if data['model'].startswith("sdxl-"):
                # multiply the width and height by the global_settings['sdxl_resolution_multiplier']:
                data['width'] = data['width'] * global_settings['sdxl_resolution_multiplier']
                data['height'] = data['height'] * global_settings['sdxl_resolution_multiplier']
                
                data['width'] = round_to_multiple_of_eight(data['width'])
                data['height'] = round_to_multiple_of_eight(data['height'])
        
        

        # Validate and preprocess the input data
        validated_data, error_message = validate_input_data(data)
        if error_message:
            return generate_error_response(error_message, 400)

        # Check for banned users and fast pass
        account_id = validated_data['accountId']
        request_type = validated_data['request_type']
        fastpass = validated_data.get('fastpass', None)
        
        if request_type == "latent_couple":
            return generate_error_response("Latent couple is currently disabled.", 503)

        if account_id is not None:
            ban_check_result = check_banned_users(account_id, request_type)
            if ban_check_result is not False:
                return generate_error_response(f"User {account_id} is banned for {ban_check_result['reason']}, in the {request_type} category", 400)
            
        if int(account_id) != 0:
            # if account_id is already in the request_queue, return an error:
            for item in request_queue:
                if int(item.data['accountId']) == int(account_id):
                    if item.status == "queued":
                        # set the generate_retries to 0 if it doesn't exist:
                        if 'generate_retries' not in item.data:
                            item.data['generate_retries'] = 0

                        if item.data['generate_retries'] == 3:
                            # cancel the request:
                            item.status = "cancelled"
                            request_queue.remove(item)
                            return generate_error_response(f"User {account_id} already has a request in the queue, the request has been cancelled", 400)
                        else:
                            item.data['generate_retries'] += 1
                            return generate_error_response(f"User {account_id} already has a request in the queue, retry {3 - item.data['generate_retries']} more times to cancel the request", 400)

                    else:
                        return generate_error_response(f"User {account_id} already has a request in the queue", 400)
            

        fastpass_enabled, error_message = check_fast_pass(fastpass, validated_data)
        if error_message:
            return generate_error_response(error_message, 400)
        
        if global_settings.get('upscale', False):
            if validated_data['upscale']:
                return generate_error_response("Upscaling is currently disabled.", 503)

        # Check if the model is valid
        model_name = validated_data['model']
        if model_name not in txt2img_models:
            return generate_error_response("Invalid model name", 400)

        # Prepare the data for the request
        data = prepare_request_data(validated_data)

        request_id = str(uuid.uuid4())
        queue_item = QueueRequest(request_id, data)

        # Check for duplicate requests
        if queue_item.data in [item.data for item in request_queue]:
            return generate_error_response("Duplicate request", 400)
        
        print(f"Fast pass: {fastpass}, Fast pass enabled: {fastpass_enabled}")

        # Add the request to the queue
        if fastpass and fastpass_enabled:
            request_queue.insert(0, queue_item)
            print(f"Fast pass {fastpass} is enabled, adding to the front of the queue...")
        else:
            request_queue.append(queue_item)

        position = len(request_queue)  # Current position in the queue is its length

        return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        print(data)
        return generate_error_response(error_message, 500)  # Return the error response within the request handler

generateTestJsonLatentCouple = {
    "model": "sonic",
    "prompt": "((Masterpiece)), HD, Best Quality, shading, intricate details, (4k, 2d, digital art, airbrush) hyper detailed, high resolution, hyper realistic, 2girls, on-left, on-right, bikini, beach AND 1girl, amy rose, bikini, beach AND 1girl, rouge the bat, bikini, beach",
    "negativeprompt": "loli, child, monochrome, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 20,
    "width": 768,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "latent_couple",
    "splitType": "horizontal",
    "splits": 2,
    "splitRatio": 0.2,
    "splitOverlap": 0,
    "strength": 0.65,
    "lora": [],
    "upscale": False
}

generateTestJson1 = {
    "model": "aing",
    "prompt": "1girl, cute, swivel",
    "negativeprompt": "worst quality, low quality",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2video",
    "lora": [],
    "video_length": 16,
    "guidance": 5,
    "upscale": False
}


generateTestJsonSDXL = {
    "model": "sdxl-ponydiffusion",
    "prompt": "score_9, score_8_up, 2d, coco bandicoot, large breasts, thick thighs, denim shorts, black crop top, in city",
    "negativeprompt": "(score_6, score_5, 3d, hyperrealistic, octane renderer, monochrome, black and white, rough sketch)",
    "steps": 20,
    "aspect_ratio": "portrait",
    "seed": 123123123,
    "quantity": 4,
    "request_type": "txt2img",
    "lora": [],
    "upscale": False
}


generateTestJson2 = {
    "model": "furryblend",
    "prompt": "1girl, coco bandicoot, thicc, beach, bikini",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 20,
    "width": 512,
    "height": 768,
    "seed": -1,
    "quantity": 1,
    "aspect_ratio": "portrait",
    "request_type": "txt2img",
    "scheduler": "dpm",  # "eulera" or "dpm"
    "lora": ['character-cocobandicoot', 'style-afrobull', 'effect-furthermore'],
    "upscale": True,
}

generateTestJson2a = {
    "model": "furryblend",
    "prompt": "1girl, coco bandicoot, thicc, beach, bikini ",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": [],
    "upscale": False,
}

generateTestJson22 = {
    "model": "realisticVision",
    "prompt": "1girl, coco bandicoot, nude, sexy,",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": ['character-cocobandicoot'],
    "upscale": False
}

generateTestJson222 = {
    "model": "furry",
    "prompt": "1girl, ochako uraraka, nude, sexy,",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": [],
    "upscale": False
}

generateTestJson3 = {
    "model": "fluffysonic",
    "prompt": "1girl, {rouge the bat, denim shorts, croptop|amy rose, bikini, beach|vanilla the bunny, apron, kitchen, window}, nude, sexy,",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 19,
    "width": 512,
    "height": 768,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": [],
    "upscale": False
}


generateTestJson4 = {
    "model": "aing",
    "prompt": "1girl",
    "steps": 5,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": []
}

generateTestJson5 = {
    "model": "flat2DAnimerge",
    "prompt": "1girl",
    "steps": 5,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": []
}

generateTestJson6 = {
    "model": "realisticVision",
    "prompt": "1girl",
    "steps": 5,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": []
}

def test_generate_image(test_data):
    with app.test_client() as client:
        response = client.post('/generate', json=test_data)
        status_code = response.status_code
        # Additional assertions or checks on the response can be added here
        print(f"Status Code: {status_code}")
        print(response.json)

# for every txt2img model, create_and_load_model:
# txt2img_models = {
#         'furry': {'loaded':None, 'model_path': furry_model_path},
#         'sonic': {'loaded':None, 'model_path': sonic_model_path},
#         'aing': {'loaded':None, 'model_path': aing_model_path},
#         'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path},
#         'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path},
#         'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path},
#     }

# use lora_weights_map to create a for loop that runs test_generate_image for each lora in the lora_weights_map:

def test_all_loras():
    
    for lora_name, lora_data in lora_weights_map.items():
        if lora_name == "background":
            for lora in lora_data:
                generateTest = {
                    "model": "fluffysonic",
                    "prompt": f"1girl, amy rose, nude, sexy",
                    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                    "seed": 24682468,
                    "accountId": 1039574722163249233,
                    "quantity": 1,
                    "request_type": "txt2img",
                    "lora": [f"{lora}"],
                    "upscale": False,
                    "loraTest": True
                }
                test_generate_image(generateTest)
                
# test_all_loras()




# test_generate_image(generateTestJsonLatentCouple)
test_generate_image(generateTestJsonSDXL)
# test_generate_image(generateTestJson1)
# test_generate_image(generateTestJson2)
# test_generate_image(generateTestJson2a)
# test_generate_image(generateTestJson22)
# test_generate_image(generateTestJson222)
# test_generate_image(generateTestJson3)



    
def run_flask_app():
    app.run(host='0.0.0.0', port=5003)

def start_background_tasks():
    threading.Thread(target=process_queue, daemon=True).start()
    threading.Thread(target=run_flask_app, daemon=True).start()

async def main():
    start_background_tasks()
    await process_hash_queue()

if __name__ == '__main__':
    asyncio.run(main())
    print(torch.__version__)
    print("Startup time: " + str(time.time() - program_start_time) + " seconds")
    app.run(host='0.0.0.0', port=5003)