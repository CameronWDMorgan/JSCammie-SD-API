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
from diffusers import AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, AnimateDiffPipeline, MotionAdapter, ControlNetModel, StableDiffusionUpscalePipeline, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline
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

device = torch.device("cuda")

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
# xl_pony_model_path = config["xl_pony_model_path"]

global lora_metadata_list
lora_metadata_list = []
try:
        
    img2img_models = {}
    
    upscale_model = {
        '4x': {'loaded':None, 'model_path': 'stabilityai/stable-diffusion-x4-upscaler', 'scheduler': EulerAncestralDiscreteScheduler}
    }
    
    inpainting_models = {
        'inpainting': {'loaded':None, 'model_path': './models/inpainting/SonicDiffusionV4-inpainting.inpainting.safetensors', 'scheduler': EulerAncestralDiscreteScheduler},
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        # 'xl-pony': {'loaded':None, 'model_path': xl_pony_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }

    txt2img_models = {
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        # 'xl-pony': {'loaded':None, 'model_path': xl_pony_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }
    
    txt2video_models = {
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }
    
    openpose_models = {
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'furryblend': {'loaded':None, 'model_path': furryblend_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'ponydiffusion': {'loaded':None, 'model_path': ponydiffusion_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        # 'xl-pony': {'loaded':None, 'model_path': xl_pony_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }
        
    # for each model in txt2img_models that doesnt have a save_pretrained folder, create one by using StableDiffusionPipeline, loading the model and using the name as the final folder:
    for model_name, model_info in txt2img_models.items():
        if not os.path.exists('./models/' + model_name):
            print("Creating folder for " + model_name)
            try:
                if model_name.startswith("xl-"):
                    pipeline = AutoPipelineForText2Image.from_single_file(
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


        

def create_and_load_inpainting_model(model_path, name, scheduler, model_type, data):
    if name.startswith("xl-"):
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            './models/' + name, 
            torch_dtype=torch.float16,
            revision="fp16",
        )
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
        
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    pipeline.load_textual_inversion("./embeddings/EasyNegativeV2.safetensors")
    pipeline.load_textual_inversion("./embeddings/BadDream.pt")
    pipeline.load_textual_inversion("./embeddings/boring_e621_v4.pt")
    pipeline.enable_model_cpu_offload()
    
    return pipeline


def create_and_load_model(model_path, name, scheduler, model_type, data):

    if name.startswith("xl-"):
        pipeline = AutoPipelineForText2Image.from_pretrained(
            './models/' + name,
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
    
    pipeline.enable_vae_slicing()
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    pipeline.load_textual_inversion("./embeddings/EasyNegativeV2.safetensors")
    pipeline.load_textual_inversion("./embeddings/BadDream.pt")
    pipeline.load_textual_inversion("./embeddings/boring_e621_v4.pt")
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_vae_tiling()
    
    components = pipeline.components
    
    if name.startswith("xl-"):
        imgpipeline = StableDiffusionXLImg2ImgPipeline(**components)
    else:
        components['safety_checker'] = None
        imgpipeline = StableDiffusionImg2ImgPipeline(**components, requires_safety_checker=False)

    img2img_models[name] = imgpipeline
    pipeline.enable_model_cpu_offload()

    return pipeline

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")

def create_and_load_txt2video_model(model_path, name, scheduler, model_type, data):
    
    pipeline = AnimateDiffPipeline.from_pretrained(
        './models/' + name,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        './models/' + name,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    
    pipeline.scheduler = scheduler
    
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    # print('\npassed attn processor')
    pipeline.load_textual_inversion("./embeddings/EasyNegativeV2.safetensors")
    # print('\npassed easy negative')
    pipeline.load_textual_inversion("./embeddings/BadDream.pt")
    # print('\npassed bad dream')
    pipeline.load_textual_inversion("./embeddings/boring_e621_v4.pt")
    # print('\npassed boring e621')
        
    # print('\npassed scheduler')
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
    
    # components = pipeline.components
    # components['safety_checker'] = None
    
    # print("Txt2Video Model Loaded")

    return pipeline


def create_and_load_controlnet_model(model_path, name, scheduler, model_type, data):
    
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
    pipeline.load_textual_inversion("./embeddings/EasyNegativeV2.safetensors")
    pipeline.load_textual_inversion("./embeddings/BadDream.pt")
    pipeline.load_textual_inversion("./embeddings/boring_e621_v4.pt")
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_vae_tiling()
    
    components = pipeline.components
    components['safety_checker'] = None
    pipeline.enable_model_cpu_offload()

    return pipeline
    
def create_and_load_upscale_model(model_path, scheduler, model_type, data):
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_path,
        revision="fp16",
    )
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    pipeline.enable_attention_slicing("max")
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    
    return pipeline
    
    
def get_upscale_model(name, data):
    global upscale_model
    model_info = upscale_model['4x']
    
    # print("Upscale Model Info: ", model_info)
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_upscale_model(model_info['model_path'], model_info['scheduler'], data['request_type'], data)
    else:
        model_info = upscale_model['4x']
    
    return model_info['loaded']
    
def get_txt2img_model(name, data):
    model_info = txt2img_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    else:
        model_info = txt2img_models[name]

                
    return model_info['loaded']

def get_txt2video_model(name, data):
    model_info = txt2video_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_txt2video_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    else:
        model_info = txt2video_models[name]

                
    return model_info['loaded']



def get_inpainting_model(name, data):
    if data['inpainting_original_option'] == False:
        name = "inpainting"
    model_info = inpainting_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_inpainting_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    else:
        model_info = inpainting_models[name]
                   
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
        model_info['loaded'] = create_and_load_controlnet_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    else:
        model_info = openpose_models[name]
        
    return model_info['loaded']




def update_lora_weights_map():
    global lora_weights_map
    while True:
        try:
            with open('./lora_weights.yaml', 'r', encoding='utf-8') as f:
                lora_weights_content = f.read()
            # print("Lora Weights Content:", lora_weights_content)  # Debugging line
            lora_weights_map = yaml.safe_load(lora_weights_content)

        except Exception as e:
            print(f"Error reading the YAML file: {e}")
        time.sleep(1)

# Start a separate thread that updates the lora_weights_map
threading.Thread(target=update_lora_weights_map, daemon=True).start()

app = Flask(__name__)
CORS(app)

@app.route('/get-lora-yaml')
def get_lora_yaml():

    return jsonify(lora_weights_map)


from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

def add_watermark(image, text, font_size):
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("./Amaranth-Regular.ttf", font_size) 
    except IOError:
        font = ImageFont.load_default()  # Load the default font in case of error
        
    # get the image size so I can calculate the offset of the watermark to be in the top right corner:
    x = image.width - 10 - font.getsize(text)[0]  # 10 pixels offset from the right edge
    y = 4  # 10 pixels offset from the top edge

    # Text outline
    outline_color = "black"
    for offset in [(1,1), (1,-1), (-1,1), (-1,-1)]:  # Offsets for the outline
        draw.text((x+offset[0], y+offset[1]), text, font=font, fill=outline_color)

    # Text itself
    text_color = "white"
    draw.text((x, y), text, font=font, fill=text_color)

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

# def load_loras(request_id, current_model, lora_items, data):
#     global lora_metadata_list
#     lora_metadata_list = []
#     for item in lora_items:
                                    
#         category, key = item.split('-', 1)
#         lora_category = lora_weights_map.get(category, {})
#         lora_data = lora_category.get(f"{category}-{key}")  # Ensure to use the full key

#         try:
#             if lora_data:
#                 strength = 1.0  
#                 # Check if 'strength' key exists in lora_data
                
#                 if category == "character":
#                     strength = 0.5
                
#                 if 'strength' in lora_data:
#                     strength = lora_data['strength']
                    
#                 print(f"Found data for {item}: {lora_data['name']} - strength: {strength}")
#                 lora_metadata = f"{lora_data['name']} - strength: {strength}"
#                 lora_metadata_list.append(lora_metadata)
#                 current_model.load_lora_weights(lora_data['lora'], low_cpu_mem_usage=False, ignore_mismatched_sizes=True, cross_attention_kwargs={"scale": strength} )
#             else:
#                 print(f"No data found for {item}")
        
#         except Exception as e:
#             results[request_id] = {"error": str(e)}
#             # convert ' to " to avoid JSON errors:
#             e = str(e).replace("'", '"')
#             if e == '"LayerNormKernelImpl" not implemented for "Half"':
#                 if data['request_type'] == 'txt2img':
#                     txt2img_models[data['model']]['loaded'] = None
#                 elif data['request_type'] == 'img2img':
#                     img2img_models[data['model']]['loaded'] = None
#                 elif data['request_type'] == 'inpainting':
#                     inpainting_models[data['model']]['loaded'] = None
#                 elif data['request_type'] == 'openpose':
#                     openpose_models[data['model']]['loaded'] = None
#                 elif data['request_type'] == 'txt2video':
#                     txt2video_models[data['model']]['loaded'] = None
#                 e = e + " | Model Reloaded"
#             print("Error processing request:", e)

# def load_loras(request_id, current_model, lora_items, data):
#     global lora_metadata_list
#     lora_metadata_list = []
    
#     # Parse the prompt for Lora settings and strengths
#     lora_settings = {}
#     import re
    
#     # Assuming data['prompt'] contains the string with Lora settings
#     prompt = data.get('prompt', '')
#     pattern = re.compile(r"(style|effect|concept|clothing|character|pose|background)-([a-zA-Z0-9]+):([0-9.]+)")
#     matches = pattern.findall(prompt)
#     for match in matches:
#         lora_key, key, strength_str = match
#         lora_settings[f"{lora_key}-{key}"] = float(strength_str)
    
#     for item in lora_items:
#         split_item = item.split('-', 1)
#         if len(split_item) < 2:
#             print(f"Error processing item '{item}': incorrect format.")
#             continue  # Skip this item and continue with the next one
#         category, key = split_item
        
#         lora_category = lora_weights_map.get(category, {})
#         lora_data = lora_category.get(item)  # Use the full item as key
        
#         try:
#             if lora_data:
#                 # Default strength
#                 strength = 1.0
                
#                 # Check if this Lora is specified in the prompt and adjust the strength accordingly
#                 if item in lora_settings:
#                     strength = lora_settings[item]
#                 elif 'strength' in lora_data:
#                     # Use the strength specified in the Lora data if not overridden by the prompt
#                     strength = lora_data['strength']
                
#                 print(f"Found data for {item}: {lora_data['name']} - strength: {strength}")
#                 lora_metadata = f"{lora_data['name']} - strength: {strength}"
#                 lora_metadata_list.append(lora_metadata)
#                 current_model.load_lora_weights(lora_data['lora'], low_cpu_mem_usage=False, ignore_mismatched_sizes=True, cross_attention_kwargs={"scale": strength})
#             else:
#                 print(f"No data found for {item}")
        
#         except Exception as e:
#             results[request_id] = {"error": str(e)}
#             e = str(e).replace("'", '"')
#             print("Error processing request:", e)



import re

# Updated pattern to include optional minus sign before the number
loraPattern = re.compile(r"(style|effect|concept|clothing|character|pose|background)-([a-zA-Z0-9]+):(-?[0-9.]+)")

def load_loras(request_id, current_model, lora_items, data):
    global lora_metadata_list
    start_time = time.time()
    lora_metadata_list = []

    # Parse the prompt for Lora settings and strengths
    prompt = data.get('prompt', '')
    lora_settings = {f"{match[0]}-{match[1]}": float(match[2]) for match in loraPattern.findall(prompt)}
    
    # Remove the matched patterns from the prompt
    cleaned_prompt = re.sub(loraPattern, '', prompt)
    data['prompt'] = cleaned_prompt.strip()  # Remove leading/trailing whitespace if any
    
    print(f"Cleaned Prompt: {data['prompt']}\n")
    
    adapter_name_list = []
    adapter_weights_list = []

    for item in lora_items:
        try:
            category, key = item.split('-', 1)
            lora_data = lora_weights_map.get(category, {}).get(item)

            if lora_data:
                strength = lora_settings.get(item, lora_data.get('strength', 1.0))

                if strength:  # This checks for strength != 0; it will work with negative numbers as well
                    lora_metadata = f"{lora_data['name']} - strength: {strength}"
                    lora_metadata_list.append(lora_metadata)

                    current_model.load_lora_weights(
                        lora_data['lora'], 
                        low_cpu_mem_usage=False, 
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
        current_model.set_adapters(adapter_name_list, adapter_weights=adapter_weights_list)
        current_model.fuse_lora()
    except Exception as e:
        print(f"Error during model configuration: {e}")
        
    print(f"Time taken to set adapters: {time.time() - set_adapters_start_time:.2f} seconds")
    










def process_image(current_model, model_type, data, request_id, save_image=False):
    try:
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
        elif model_type == 'img2img':
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
            elif data['request_type'] == 'txt2video':
                txt2video_models[data['model']]['loaded'] = None
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
        # Load upscale settings once if needed and adjust font-size accordingly
        if data.get('upscale', False):
            with open('upscale-settings.yaml', 'r') as file:
                upscaleSettings = yaml.safe_load(file)
            data['font-size'] = upscaleSettings.get('font-size', 24)
        else:
            data['font-size'] = 24

        # Simplify accountId handling
        accountId_string = "" if data.get('accountId') == "0" else data.get('accountId', '')

        if data.get('upscale', False) and model_type != "txt2video":
            model = get_upscale_model('aaa', data)
            steps = upscaleSettings.get('steps', 0)
            
            # Resizing the image for upscaling
            image_width = round((output_image.width / 1.25) / 8) * 8
            image_height = round((output_image.height / 1.25) / 8) * 8
            output_image = output_image.resize((image_width, image_height))

            output_image.save("og_image.png")

            # Perform upscaling with model
            try:
                with torch.inference_mode():
                    output_image = model(
                        prompt=data['prompt'],
                        negative_prompt=data['negative_prompt'],
                        image=output_image,
                        generator=data['seed'],
                        num_inference_steps=steps,
                    ).images[0]
            except Exception as e:
                print(f"Error upscaling image: {e}")

            # Clear torch memory after upscaling
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Generate metadata
        metadata = {
            "request_id": request_id,
            "model_type": model_type,
            "prompt": data['true_prompt'],
            "loras": [],  # Placeholder for lora_metadata_list, assuming it's defined elsewhere
            "steps": data['steps'],
            "CFG": data['guidance'],
            "model": data['model'],
            "upscaled": data['upscale'],
            "generation_date": datetime.datetime.now().isoformat(),
            "accountId": str(data['accountId'])
        }

        # Add watermark, if applicable
        if model_type != "txt2video":
            watermarked_image = add_watermark(output_image, "JSCammie.com", font_size)
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
        
        # print(data['upscale'])  
        
        if data['strength'] > 1:
            data['strength'] = data['strength'] / 100
        
        if data['strength'] < 0:
            data['strength'] = 0.01

        if data['image_data'] is not None:
            try:
                img_bytes = io.BytesIO(data['image_data'])
                data['image_data'] = Image.open(img_bytes)
                
                # Determine the scaling factor to ensure both sides are at least 512px
                scale_factor = max(512 / data['image_data'].width, 512 / data['image_data'].height)

                # Calculate new dimensions
                new_width = round_to_multiple_of_eight(data['image_data'].width * scale_factor)
                new_height = round_to_multiple_of_eight(data['image_data'].height * scale_factor)

                # Update dimensions in the data dictionary
                data['width'], data['height'] = new_width, new_height

                # Resize the image
                data['image_data'] = data['image_data'].resize((new_width, new_height), Image.ANTIALIAS)
                data['image_data'] = data['image_data'].convert('RGB')

            except Exception as e:
                error_message = "Failed to identify image file"
                print("Error processing request:", error_message)
                results[request_id] = {"status": "error", "message": error_message}
                queue_item.status = "error"
                return "skipped"
        else:
            data['image_data'] = None
            
            
            
        if data['mask_data'] is not None:
            try:
                img_bytes = io.BytesIO(data['mask_data'])
                data['mask_data'] = Image.open(img_bytes)
                data['mask_data'] = data['mask_data'].resize((round(data['width']), round(data['height'])))
                data['mask_data'].save("inpaintingBefore.png")
            except Exception as e:
                error_message = "Failed to identify image file"
                print("Error processing request:", error_message)
                results[request_id] = {"status": "error", "message": error_message}
                queue_item.status = "error"
                return "skipped"
        else:
            data['mask_data'] = None
            
        # if data['request_type'] == 'controlnet_img2img':
        #     image = np.array(data['image_data'])
        #     low_threshold = 100
        #     high_threshold = 200
        #     image = cv2.Canny(image, low_threshold, high_threshold)
        #     image = image[:, :, None]
        #     image = np.concatenate([image, image, image], axis=2)
        #     canny_image = Image.fromarray(image)
        #     data['image_data'] = canny_image

            
        if data['steps'] > 70:
            if data['image_count'] > 6:
                data['image_count'] = 1
            
            
            
        if data['seed'] == -1:
            data['seedNumber'] = random.randint(0, 2**32 - 1)
        else:
            data['seedNumber'] = data['seed']
        
        promptString = str(data['prompt'])
        
        # if prompt is a list then join it into a string:
        if isinstance(data['prompt'], list):
            promptString = ' '.join(data['prompt'])
        
        # data on multiple print lines for easier debugging
        print("Request Type: " + str(data['request_type']) + "\nModel: " + str(data['model']) + " | Steps: " + str(data['steps']) + " | Width: " + str(data['width']) + "px | Height: " + str(data['height']) + "px\nSeed: " + str(data['seedNumber']) + " | Strength: " + str(data['strength']) + " | CFGuidance: " + str(data['guidance']) + " | Image Count: " + str(data['image_count'] )  + "\nPrompt: " + str(promptString) + "\nNegative Prompt: " + str(data['negative_prompt']) + "\nLora: " + str(data['lora']))

        model_name = data['model']
        lora = data.get('lora', "NO")
        model_type = data['request_type']
        
                
        # if model_type is txt2img or img2img, get the model, else get the inpainting model:
        if model_type == 'txt2img' or model_type == 'img2img':
            model = get_txt2img_model(model_name, data)
        elif model_type == 'inpainting':
            model = get_inpainting_model(model_name, data)
        # elif model_type == 'controlnet_img2img':
        #     model = get_controlnet_img2img_model(model_name, data)
        elif model_type == 'openpose':
            model = get_openpose_model(model_name, data)
        elif model_type == 'latent_couple':
            model = get_txt2img_model(model_name, data)
            data['inpainting_original_option'] = True
            inpainting_model = get_inpainting_model(model_name, data)
            img2img_model = img2img_models.get(model_name)
        elif model_type == 'txt2video':
            model = get_txt2video_model(model_name, data)
            
            
            
        # checks the model type to load the correct model:
        if model_type != 'img2img':
            current_model = model
        else:
            current_model = img2img_models.get(model_name)
       
       
       
            
        if current_model is not None:
            current_model.unfuse_lora()
            current_model.unload_lora_weights()
            print(model_type)
            if data['lora'] is not None and model_type != 'latent_couple':
                load_loras(request_id, current_model, lora, data)
            if model_type != 'latent_couple':
                model_outputs = process_image(current_model, model_type, data, request_id)
            else:
                inpainting_model.unfuse_lora()
                inpainting_model.unload_lora_weights()
                print("Latent Couple Generation Process Started")
                
                prompt_og = data['prompt']
                
                data['prompt'] = prompt_og[0]
                
                load_loras(request_id, inpainting_model, lora, data)
                
                base_images = process_image(current_model, 'txt2img', data, request_id)  
                
                for i, base_image in enumerate(base_images):
                    base_image.save(f"base{i}.png")
            
                if data['steps'] > 50:
                    data['steps'] = 50
                               
                slices = []
                
                base_image = base_images[0]

                for i in range(data['splits']):
                    if data['splitType'] == "horizontal":
                        slice_width = base_image.width / data['splits']
                        overlap_width = slice_width * data['splitOverlap']
                        slice_width = round_to_multiple_of_eight(slice_width + overlap_width)

                        left = i * (base_image.width / data['splits']) - (overlap_width if i > 0 else 0)
                        right = left + slice_width
                        sliceImageMask = Image.new('RGB', (int(base_image.width), int(base_image.height)), (0, 0, 0))
                        round_to_multiple_of_eight(left)
                        round_to_multiple_of_eight(right)
                        sliceImageMask.paste((255, 255, 255), (int(left), 0, int(right), int(base_image.height)))
                        slices.append(sliceImageMask)
                    else:  # for vertical split
                        slice_height = base_image.height / data['splits']
                        overlap_height = slice_height * data['splitOverlap']
                        slice_height = round_to_multiple_of_eight(slice_height + overlap_height)

                        top = i * (base_image.height / data['splits']) - (overlap_height if i > 0 else 0)
                        bottom = top + slice_height
                        
                        sliceImageMask = Image.new('RGB', (int(base_image.width), int(base_image.height)), (0, 0, 0))
                        round_to_multiple_of_eight(top)
                        round_to_multiple_of_eight(bottom)
                        sliceImageMask.paste((255, 255, 255), (0, int(top), int(base_image.width), int(bottom)))
                        slices.append(sliceImageMask)
                                                        
                model_outputs = []
                        
                # Save all the mask slices with random names and process each base image with each mask
                for i, mask_slice in enumerate(slices):
                    mask_slice.save(f"slice{i}.png")
                    data['mask_data'] = mask_slice
                    
                    data['prompt'] = prompt_og[i + 1]
                    print("Prompt: ", data['prompt'])
                    
                    inpainting_model.unfuse_lora()
                    inpainting_model.unload_lora_weights()
                    
                    load_loras(request_id, inpainting_model, lora, data)

                    processed_images_for_this_mask = []
                    for j, base_image in enumerate(base_images):
                        data['image_data'] = base_image
                        output = process_image(inpainting_model, 'inpainting', data, request_id)
                        
                        output[0].save(f"processed_slice{j}-{i}.png")

                        processed_image = output[0] if isinstance(output, list) else output
                        processed_images_for_this_mask.append(processed_image)

                    # Update base_images for the next iteration of mask slices
                    base_images = processed_images_for_this_mask

                # Final model outputs after processing with all masks
                model_outputs = base_images
                
                
                
                data['image_data'] = model_outputs
                
                promptString = ""
                
                for prompt in prompt_og:
                    promptString += prompt + " "

                with open('latent-loopback-settings.yaml', 'r') as file:
                    loopback_data = yaml.safe_load(file)
                    
                if loopback_data['prompt_override'] is not None:
                    data['prompt'] = prompt_og[0]
                else:
                    data['prompt'] = promptString

                # Access the data
                data['steps'] = loopback_data['steps']
                data['strength'] = loopback_data['strength']
                
                if loopback_data['enabled'] is True:
                    img2img_model.unfuse_lora()
                    img2img_model.unload_lora_weights()
                    load_loras(request_id, img2img_model, lora, data)
                    model_outputs = process_image(img2img_model, 'img2img', data, request_id)
                
                # get the type of model_outputs:
                
                promptString = ""
                
                for prompt in prompt_og:
                    promptString += prompt + " "

                data['prompt'] = promptString       
                data['image_data'] = None      

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
            if model_type == 'latent_couple':
                if data['lora'] is not None:
                    inpainting_model.unfuse_lora()
                    inpainting_model.unload_lora_weights()
                    if loopback_data['enabled'] is True:
                        img2img_model.unfuse_lora()
                        img2img_model.unload_lora_weights()
                        
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if model_outputs is not None:
                image_data_list = []

                if model_type == 'txt2video':
                    
                    all_frames = model_outputs.frames[0]
                    processed_frames = []
                    for img in all_frames:
                        frame = add_watermark(img, "JSCammie.com", 24)
                        processed_frames.append(frame)
                        
                    # Export the processed PIL images to an MP4 file
                    data['video_string'] = export_to_mp4(processed_frames, "animation.mp4")
                    
                    model_outputs = [all_frames[0]]
                    
                timeBeforeSave = time.time()
                                    
                for index, img in enumerate(model_outputs):
                    image_data = save_image(request_id, img, model_type, data, index)
                    image_data_list.append(image_data)
                    
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
    for index, item in enumerate(request_queue):
        print("Queue Status: " + item.status + " | Request ID: " + item.request_id)
    print("Queue Length: " + str(len(request_queue)) + "\n")    
        
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














def process_queue():
    global processor_busy  # Use the global keyword to modify the global variable
    
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

def check_fast_pass(fastpass, data):
    print(f"Checking fast pass: {fastpass} against the fast pass map: {fast_passes_map}")
    
    if str(fastpass) == str(0):
        return False
    
    # Ensure thread-safe read
    with threading.Lock():
        fastpass_data = fast_passes_map.get('passes', {}).get(fastpass, {})
        
        # Check if the fastpass is enabled
        if not fastpass_data.get('enabled', False):
            return "error1"
        
        print(f"fastpass discordId: {fastpass_data.get('discordId')}, data accountId: {data['accountId']}")

        # Check if the discordId matches
        if int(fastpass_data.get('discordId')) != int(data['accountId']):
            return "error2"

        # Check if the fast pass has not expired
        if time.time() > fastpass_data.get('expires', 0):
            return "error3"

        print(f"Fast pass {fastpass} is valid and active.")
        return True
    
    
    
    
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
        
# Start a separate thread that updates the lora_weights_map
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

                
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        
        true_prompt = data['prompt']
        
        data['prompt'] = data['prompt'].replace('\r', '').replace('\n', '')
        
        data['accountId'] = data.get('accountId', 0)
                
        if data['accountId'] == "":
            data['accountId'] = 0
        
        data['accountId'] = int(data['accountId'])
        
        data['prompt'] = randomize_string(data['prompt'])
        
        if str(data['prompt']) == "{'status': 'error', 'message': 'Mismatched brackets'}":
            error_message = "Mismatched brackets ('{}' brackets are used to denote a random choice, and must be used in pairs, here is an example of a correct usage: '{woman|man} with {long|short} hair')"
            print(error_message)
            return generate_error_response(error_message, 400)
        
        if int(data['width']) > 512 and int(data['height']) > 512:
            if data['request_type'] == "txt2video":
                data['width'] = 512
                data['height'] = 512
            elif not data['model'].startswith("xl-"):
                data['steps'] = 20
                
        if data['accountId'] is not None:
            result = check_banned_users(data['accountId'], data['request_type'])
            if result is not False:
                error_message = f"User {data['accountId']} is banned for {result['reason']}, in the {data['request_type']} category"
                print(error_message)
                return generate_error_response(error_message, 400)
            
        fastpass = data.get('fastpass', None)
            
        fastpass_enabled = check_fast_pass(fastpass, data)
        
        if fastpass_enabled == "error1":
            error_message = "Fast pass is not enabled."
            print(error_message)
            return generate_error_response(error_message, 400)
        elif fastpass_enabled == "error2":
            error_message = "Discord ID does not match the fast pass."
            print(error_message)
            return generate_error_response(error_message, 400)
        elif fastpass_enabled == "error3":
            error_message = "Fast pass has expired."
            print(error_message)
            return generate_error_response(error_message, 400)
            
        # if data['model'] value isnt a key in the txt2img_models dictionary, return an error:
        if data['model'] not in txt2img_models:
            error_message = "Invalid model name"
            print(error_message)
            return generate_error_response(error_message, 400)
        
        

        # Extract the image data from the request and store it
        input_image = data.get('image', None)
        if input_image:
            if input_image.startswith('data:image'):
                input_image = input_image.split(',')[1]
            input_image += '=' * (-len(input_image) % 4)  # Pad if necessary
            image_data = base64.b64decode(input_image)
        else:
            image_data = None
            
        input_mask = data.get('inpaintingMask', None)
        if input_mask:
            if input_mask.startswith('data:image'):
                input_mask = input_mask.split(',')[1]
            input_mask += '=' * (-len(input_mask) % 4)  # Pad if necessary
            mask_data = base64.b64decode(input_mask)
        else:
            mask_data = None
            
            
        print(f"{data['request_type']} \n")
            
        
            
        # append mask_data and image_data to data
        data['image_data'] = image_data
        data['mask_data'] = mask_data
                
        # validation checks:
        
        upscale = data.get('upscale', None)
        
        print(f"Upscale: {upscale}")
        
        if upscale != False:
            error_message = "Upscaling is temporarily disabled! Sorry for the inconvenience."
            print(error_message)
            return generate_error_response(error_message, 400)
        
        if upscale and data['quantity'] > 1:
            error_message = "You can only generate one image at a time when using the upscale option!"
            print(error_message)
            return generate_error_response(error_message, 400)
        if not data['model'].startswith("xl-"):
            if data['height'] > 768 or data['width'] > 768:
                error_message = "Image dimensions are too large. Please use an image with a maximum resolution of 768x768."
                print(error_message)
                return generate_error_response(error_message, 400)
        
        if data['steps'] > 100:
            error_message = "You have reached the limit of 100 steps per request. Please reduce the number of steps and try again."
            print(error_message)
            return generate_error_response(error_message, 400)
        
        if data['quantity'] > 8:
            error_message = "You have reached the limit of 8 images per request. Please reduce the number of images and try again."
            print(error_message)
            return generate_error_response(error_message, 400)
                
        if data['steps'] > 20 and data['quantity'] > 8:
            error_message = "You have reached the limit of 8 images per request. Please reduce the number of steps and try again."
            print(error_message)
            return generate_error_response(error_message, 400)
        
        if len(data['lora']) > 5:
            error_message = "You have reached the limit of 5 Lora options. Please deselect some and try again."
            print(error_message)
            return generate_error_response(error_message, 400)
        
        if data['seed'] is None:
            data['seed'] = -1
            
            # if seed is less than -1 or greater than 2**32 - 1, return an error:
        if int(data['seed']) > 2**32 - 1:
            error_message = "Seed is too large. Please use a seed between -1 and 4294967295."
            print(error_message)
            return generate_error_response(error_message, 400)
        
        if int(data['seed']) < -1:
            error_message = "Seed is too small. Please use a seed between -1 and 4294967295."
            print(error_message)
            return generate_error_response(error_message, 400)
                
                
                
                
                
        if data['request_type'] == 'txt2video':
            if int(data['video_length']) > 16 or int(data['video_length']) < 6:
                error_message = "Video length is too long/short. Please use a video length between 6 and 16 frames."
                print(error_message)
                return generate_error_response(error_message, 400) 
            
            
                    
        if data['request_type'] == 'latent_couple':
            print("Latent Couple Request")
            if 'AND' in data['prompt']:
                # Split the string at each 'AND' and remove empty strings that might result from consecutive 'AND's
                data['prompt'] = [s.strip() for s in data['prompt'].split('AND') if s.strip()]
            else:
                error_message = "When using latent couple you need to use the keyword 'AND' to separate the different things you want in the scene"
                print(error_message)
                return generate_error_response(error_message, 400)
            
            # check if data['splits] is an int and if it is between 1-3:
            splits = data.get('splits', None)
            if splits is not None:
                splits = int(splits)
                data['splits'] = splits
                if splits < 1 or splits > 5:
                    error_message = "Splits needs to be between 1 and 5"
                    print(error_message)
                    return generate_error_response(error_message, 400)
            else:
                error_message = "You need to specify a number of splits"
                print(error_message)
                return generate_error_response(error_message, 400)
            
            splitType = data.get('splitType', None)
            if splitType is not None:
                if splitType != "horizontal" and splitType != "vertical":
                    error_message = "Split direction needs to be either horizontal or vertical"
                    print(error_message)
                    return generate_error_response(error_message, 400)
            else:
                error_message = "You need to specify a split direction"
                print(error_message)
                return generate_error_response(error_message, 400)
            
            if len(data['prompt']) != splits + 1:
                error_message = "You need to specify a prompt for each split by using the 'AND' keyword"
                print(error_message)
                return generate_error_response(error_message, 400)
            
            
            
            # have a loop that checks the stepsAfterStrengthApply and if it is less than 1, add 1 to data['steps'] and then check again until it is greater than 1:
            if float(data['strength']) != 1:
                if data['steps'] - (int(data['steps']) * float((data['strength']))) < 1:
                    while data['steps'] - (int(data['steps']) * float((data['strength']))) < 1:
                        data['steps'] += 1
                        
                    data['steps'] = data['steps'] * 2
                    data['steps'] += 1
                    
                    round(data['steps'])
            
        negativePromptFinal = "EasyNegativeV2, boring_e621_v4, " + data.get("negativeprompt", "")
                
        data = {
            "model": data.get('model'),
            "prompt": data.get('prompt'),
            "negative_prompt": negativePromptFinal,
            "image_count": int(data.get("quantity")),
            "steps": int(data.get("steps", 20)),
            "width": int(data.get("width", 512)),
            "height": int(data.get("height", 512)),
            "seed": int(data.get("seed", -1)),
            "strength": float(data.get("strength", 0.75)),
            "guidance": float(data.get("guidance", 5)),
            "image_data": data.get("image_data", None),
            "mask_data": data.get("mask_data", None),
            "lora": data.get('lora', None),
            "enhance_prompt": data.get('enhance_prompt', False),
            "request_type": data['request_type'],
            "upscale": data.get('upscale', False),
            "inpainting_original_option": data.get('inpainting_original_option', False),
            "splitType": data.get('splitType', "horizontal"),
            "splits": int(data.get('splits', 1)),
            "splitOverlap": float(data.get('splitOverlap', 0.1)),
            "finalStrength": float(data.get('finalStrength', 0.2)),
            "video_length": int(data.get('video_length', 16)),
            "accountId": int(data.get('accountId', 0)),
            "true_prompt": str(true_prompt),
        } 

            
        
        request_id = str(uuid.uuid4())
        queue_item = QueueRequest(request_id, data)
        
        # if the request is the exact same as another request (apart from the request_id), remove it, else add it to the queue:
        for index, item in enumerate(request_queue):
            if item.data == queue_item.data:
                error_message = "Duplicate request"
                print(error_message)
                return generate_error_response(error_message, 400)
    
        else:
            if fastpass:
                
                if fastpass_enabled == True:
                    print(f"Fast pass {fastpass} is enabled, adding to the front of the queue...")
                    request_queue.insert(0, queue_item)
                else:
                    request_queue.append(queue_item)

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
    "prompt": "2girls, sexy, side by side, facing viewer, looking at viewer, beach, bikini style-afrobull:0 style-diives:0 AND 1girl, rouge the bat, sexy, bikini style-afrobull:0 AND 1girl, amy rose, sexy, bikini style-diives:0",
    "negativeprompt": "loli, child, monochrome, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 100,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "latent_couple",
    "splitType": "horizontal",
    "splits": 2,
    "splitRatio": 0.2,
    "splitOverlap": 0,
    "strength": 0.75,
    "lora": ['effect-furtasticdetailer','style-afrobull', 'style-diives'],
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
    "model": "xl-pony",
    "prompt": "1girl, amy rose, sexy,",
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


generateTestJson2 = {
    "model": "furryblend",
    "prompt": "1girl, coco bandicoot, nude, sexy,",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": ['character-cocobandicoot', 'style-theotherhalf'],
    "upscale": False,
    "accountId": 1039574722163249233,
    "fastpass": "0"
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
    with app.test_request_context('/generate', method='POST', json=test_data):
        response, status_code = generate_image()
        print("Response:", response.get_json())
        print("Status Code:", status_code)

# for every txt2img model, create_and_load_model:
# txt2img_models = {
#         'furry': {'loaded':None, 'model_path': furry_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
#         'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
#         'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
#         'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
#         'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
#         'fluffysonic': {'loaded':None, 'model_path': fluffysonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
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
# test_generate_image(generateTestJsonSDXL)
# test_generate_image(generateTestJson1)
test_generate_image(generateTestJson2)
# test_generate_image(generateTestJson22)
# test_generate_image(generateTestJson222)
# test_generate_image(generateTestJson3)


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

if __name__ == '__main__':
    print("Starting Flask app...")
    print(torch.__version__)
    print("Startup time: " + str(time.time() - program_start_time) + " seconds")
    app.run(host='0.0.0.0', port=5003)