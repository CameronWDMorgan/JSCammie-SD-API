import base64
import math
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

print(f"Number of CUDA devices: {torch.cuda.device_count()}")

print(f"Clearing CUDA cache")

# print memory usage before clearing cache:
print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
print(f"Max Memory Cached: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")

model_move_manual = True

# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.reset_max_memory_cached()

import tomesd
from diffusers import AutoencoderKL, AutoPipelineForInpainting, AutoPipelineForImage2Image, AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, AnimateDiffPipeline, MotionAdapter, ControlNetModel, StableDiffusionUpscalePipeline, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline
from diffusers.utils import load_image, export_to_gif
from diffusers.models.attention_processor import AttnProcessor2_0
import time
from io import BytesIO
import accelerate

from compel import Compel, ReturnedEmbeddingsType

from moviepy.editor import ImageSequenceClip


from concurrent.futures import ThreadPoolExecutor

from controlnet_aux import OpenposeDetector

import numpy as np

import os
import datetime
import asyncio

import DB


global hash_queue_busy
global hash_queue

hash_queue = []

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True "
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

program_start_time = time.time()

processor_busy_0 = False

processor_queue_0_last = None
processor_queue_0_next = None



processor_busy_1 = False

processor_queue_1_last = None
processor_queue_1_next = None

def generate_error_response(message, status_code):
    response = jsonify({"status": "error", 'message': message})
    response.status_code = status_code
    return response

# Configuration Loading
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

def log_error(error_message, data=None):
    # add the error to a log file, with the date as the filename:
    # ensure the logs folder exists:
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    with open(f"logs/{datetime.datetime.now().strftime('%Y-%m-%d')}.log", "a", encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n {data}\n\n")
        
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

average_queue_times = []

try:
    
    # Model Loading
    
    inpainting_models = {
        'sonicdiffusion': {'loaded':None, 'model_path': config['sonicdiffusion_model_path']},
        'realisticVision': {'loaded':None, 'model_path': config["realisticVision_model_path"], 'helper':None},
        'fluffysonic': {'loaded':None, 'model_path': config["fluffysonic_model_path"]},
        'furryblend': {'loaded':None, 'model_path': config["furryblend_model_path"]},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': config["sdxl_ponydiffusion_model_path"]},
        'sdxl-autismmix': {'loaded':None, 'model_path': config["sdxl_autismmix_model_path"]},
        'sdxl-zonkey': {'loaded':None, 'model_path': config["sdxl_zonkey_model_path"]},
        'sdxl-sonicdiffusion': {'loaded':None, 'model_path': config["sdxl_sonicdiffusion_model_path"]},
    }

    txt2img_models = {
        'sonicdiffusion': {'loaded':None, 'model_path': config['sonicdiffusion_model_path']},
        'realisticVision': {'loaded':None, 'model_path': config["realisticVision_model_path"], 'helper':None},
        'fluffysonic': {'loaded':None, 'model_path': config["fluffysonic_model_path"]},
        'furryblend': {'loaded':None, 'model_path': config["furryblend_model_path"]},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': config["sdxl_ponydiffusion_model_path"]},
        'sdxl-autismmix': {'loaded':None, 'model_path': config["sdxl_autismmix_model_path"]},
        'sdxl-zonkey': {'loaded':None, 'model_path': config["sdxl_zonkey_model_path"]},
        'sdxl-sonicdiffusion': {'loaded':None, 'model_path': config["sdxl_sonicdiffusion_model_path"]},
    }
    
    img2img_models = {
        'sonicdiffusion': {'loaded':None, 'model_path': config['sonicdiffusion_model_path']},
        'realisticVision': {'loaded':None, 'model_path': config["realisticVision_model_path"], 'helper':None},
        'fluffysonic': {'loaded':None, 'model_path': config["fluffysonic_model_path"]},
        'furryblend': {'loaded':None, 'model_path': config["furryblend_model_path"]},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': config["sdxl_ponydiffusion_model_path"]},
        'sdxl-autismmix': {'loaded':None, 'model_path': config["sdxl_autismmix_model_path"]},
        'sdxl-zonkey': {'loaded':None, 'model_path': config["sdxl_zonkey_model_path"]},
        'sdxl-sonicdiffusion': {'loaded':None, 'model_path': config["sdxl_sonicdiffusion_model_path"]},
    }
    
    txt2video_models = {
        'sonicdiffusion': {'loaded':None, 'model_path': config['sonicdiffusion_model_path']},
        'realisticVision': {'loaded':None, 'model_path': config["realisticVision_model_path"], 'helper':None},
        'fluffysonic': {'loaded':None, 'model_path': config["fluffysonic_model_path"]},
        'furryblend': {'loaded':None, 'model_path': config["furryblend_model_path"]},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': config["sdxl_ponydiffusion_model_path"]},
        'sdxl-autismmix': {'loaded':None, 'model_path': config["sdxl_autismmix_model_path"]},
        'sdxl-zonkey': {'loaded':None, 'model_path': config["sdxl_zonkey_model_path"]},
        'sdxl-sonicdiffusion': {'loaded':None, 'model_path': config["sdxl_sonicdiffusion_model_path"]},
    }
    
    openpose_models = {
        'sonicdiffusion': {'loaded':None, 'model_path': config['sonicdiffusion_model_path']},
        'realisticVision': {'loaded':None, 'model_path': config["realisticVision_model_path"], 'helper':None},
        'fluffysonic': {'loaded':None, 'model_path': config["fluffysonic_model_path"]},
        'furryblend': {'loaded':None, 'model_path': config["furryblend_model_path"]},
        'sdxl-ponydiffusion': {'loaded':None, 'model_path': config["sdxl_ponydiffusion_model_path"]},
        'sdxl-autismmix': {'loaded':None, 'model_path': config["sdxl_autismmix_model_path"]},
        'sdxl-zonkey': {'loaded':None, 'model_path': config["sdxl_zonkey_model_path"]},
        'sdxl-sonicdiffusion': {'loaded':None, 'model_path': config["sdxl_sonicdiffusion_model_path"]},
    }
    
    # pre-load all the models on startup:
    
    # model_info = txt2img_models[name]
    
    # if model_info['loaded'] is None:
    #     model_info['loaded'] = load_models.txt2img(name, data, model_info['model_path'])
        
    # for each model in txt2img_models that doesnt have a save_pretrained folder, create one by using StableDiffusionPipeline, loading the model and using the name as the final folder:
    # for model_name, model_info in txt2img_models.items():
    #     if not os.path.exists('./models/' + model_name):
    #         print("Creating folder for " + model_name)
    #         try:
    #             if model_name.startswith("sdxl-"):
    #                 pipeline = StableDiffusionXLPipeline.from_single_file(
    #                     model_info['model_path'],
    #                     torch_dtype=torch.float16,
    #                     revision="fp16",
    #                     # feature_extractor=None,
    #                     # requires_safety_checker=False,
    #                     # cache_dir="",
    #                     # load_safety_checker=False,
    #                 )
    #                 pipeline.save_pretrained('./models/' + model_name)
    #             else:
    #                 pipeline = StableDiffusionPipeline.from_single_file(
    #                     model_info['model_path'],
    #                     torch_dtype=torch.float16,
    #                     revision="fp16",
    #                     safety_checker=None,
    #                     feature_extractor=None,
    #                     requires_safety_checker=False,
    #                     cache_dir="",
    #                     load_safety_checker=False,
    #                 )
    #                 pipeline.save_pretrained('./models/' + model_name)
    #         except Exception as e:
    #             print(f"Failed to load the model: {e}")
    #             raise

    # updated model preloading and saving:
    # for model_name, model_info in txt2img_models.items():
    #     if not os.path.exists('./models/saved/' + model_name):
    #         print("Creating folder for " + model_name)
    #         try:
    #             if model_name.startswith("sdxl-"):
    #                 pipeline = StableDiffusionXLPipeline.from_single_file(
    #                     model_info['model_path'],
    #                     torch_dtype=torch.float16,
    #                     revision="fp16",
    #                 )
    #                 pipeline.save_pretrained('./models/saved/' + model_name)
    #             else:
    #                 pipeline = StableDiffusionPipeline.from_single_file(
    #                     model_info['model_path'],
    #                     torch_dtype=torch.float16,
    #                     revision="fp16",
    #                     safety_checker=None,
    #                 )
    #                 pipeline.save_pretrained('./models/saved/' + model_name)

    #             # delete the pipeline to free up memory and clear caches:
    #             del pipeline
    #             torch.cuda.empty_cache()
    #             torch.cuda.reset_peak_memory_stats()
    #             torch.cuda.reset_max_memory_allocated()
    #             torch.cuda.reset_max_memory_cached()

    #         except Exception as e:
    #             print(f"Failed to load the model: {e}")
    #             raise
                
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    import load_models
    import get_models

except Exception as e:
    raise


    









from pathlib import Path

def load_yaml():
    global lora_weights_map
    try:
        with yaml_file_path.open("r", encoding="utf-8") as f:
            lora_weights_content = f.read()
        lora_weights_map = yaml.safe_load(lora_weights_content)
    except Exception as e:
        log_error("Error loading the YAML file", e)

def update_lora_weights_map():
    global lora_weights_map
    global yaml_file_path
    yaml_file_path = Path("lora_weights.yaml")

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
            log_error("Error monitoring the YAML file", e)
        time.sleep(5)

# Start the thread to update the lora_weights_map
threading.Thread(target=update_lora_weights_map, daemon=True).start()

app = Flask(__name__)
CORS(app)

from flask import jsonify

@app.route('/get-lora-yaml', methods=['GET'])
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
            new_text_image_data.append((0, 0, 0, 170))
        if item[3] != 0 and item[2] == 255:
            new_text_image_data.append((255, 255, 255, 170))
            
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

request_queue_0 = []  # Use a list instead of the queue module for more control
request_queue_1 = []  # Use a list instead of the queue module for more control
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
    
    start_time = time.time()
    
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
        log_error(f"Error parsing prompt for Lora settings: {e}", data)
        lora_settings = {}

    
    
    # print(f"Cleaned Prompt: {data['prompt']}\n")
    
    adapter_name_list = []
    adapter_weights_list = []

    for item in lora_items:
        time.sleep(0.01)
        try:
            category, key = item.split('-', 1)
            lora_data = lora_weights_map.get(category, {}).get(item)

            if lora_data:
                print(f"data['lora_strengths'] = {data['lora_strengths']}")
                if data['lora_strengths'] and len(data['lora_strengths']) == len(lora_items):
                    # print(f"data['lora_strengths'] = {data['lora_strengths']}")
                    # set the strength to the value in the data['lora_strengths'] list if it exists, else set it to the value in the lora_data dict:
                    lora_strengths_index = lora_items.index(item)
                    strength = data['lora_strengths'][lora_strengths_index]
                else:
                    strength = lora_settings.get(item, lora_data.get('strength', 1.0))
                    data['lora_strengths'].append(strength)
                
                
                    
                # print(f"Strength for {item}: {strength}")

                if strength:  # This checks for strength != 0; it will work with negative numbers as well

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
                log_error(f"No data found for {item}", data)
        except Exception as e:
            print(f"Error processing item '{item}': {e}")
            log_error(f"Error processing item '{item}': {e}", data)
            
    # print(f"Time taken to fetch loras: {time.time() - start_time:.2f} seconds")
        
    set_adapters_start_time = time.time()

    try:
        print(f"current_model.set_adapters({adapter_name_list}, adapter_weights={adapter_weights_list})")
        log_error(f"current_model.set_adapters({adapter_name_list}, adapter_weights={adapter_weights_list})", data)
        current_model.set_adapters(adapter_name_list, adapter_weights=adapter_weights_list)
        current_model.fuse_lora()
        
        # print(f"Time taken to set adapters: {time.time() - set_adapters_start_time:.2f} seconds")
        
        return current_model
    
    
    except Exception as e:
        print(f"Error during model configuration: {e}")
        log_error(f"Error during model configuration: {e}", data)

def process_image(current_model, model_type, data, request_id):
    try:
        
        # check if the model is on the GPU or CPU and move it to the correct device:
        
        generator = torch.Generator(device="cuda:1")

        # generator = torch.Generator(device=f"cuda:{data['gpu_id']}")
        # if data['gpu_id'] == 1:
        #     generator = torch.Generator("cuda:1")
        # else:
        #     generator = torch.Generator("cuda:0")
        
        data['seed'] = generator.manual_seed(data['seedNumber'])
        
        with torch.no_grad():
            if data['model'].startswith("sdxl-"):
                compel = Compel(tokenizer=[current_model.tokenizer, current_model.tokenizer_2] , text_encoder=[current_model.text_encoder, current_model.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True], truncate_long_prompts=False)
                conditioning, pooled = compel(data['prompt'])
                negative_conditioning, negative_pooled = compel(data['negative_prompt'])
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
            else:
                compel = Compel(tokenizer=current_model.tokenizer, text_encoder=current_model.text_encoder, truncate_long_prompts=False)
                conditioning = compel(data['prompt'])
                negative_conditioning = compel(data['negative_prompt'])
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        
        if model_type == 'txt2img':
            if data['model'].startswith("sdxl-"):
                with torch.no_grad():
                    outputs = current_model(
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_prompt_embeds=negative_conditioning,
                        negative_pooled_prompt_embeds=negative_pooled,
                        num_inference_steps=data['steps'],
                        width=data['width'],
                        height=data['height'],
                        guidance_scale=data['guidance'],
                        generator=data['seed'],
                        strength=data['strength'],
                        num_images_per_prompt=data['image_count'],
                    ).images
            else:
                with torch.no_grad():
                    outputs = current_model(
                        prompt_embeds=conditioning,
                        negative_prompt_embeds=negative_conditioning,
                        num_inference_steps=data['steps'],
                        width=data['width'],
                        height=data['height'],
                        guidance_scale=data['guidance'],
                        generator=data['seed'],
                        strength=data['strength'],
                        num_images_per_prompt=data['image_count'],
                    ).images
        elif model_type == 'img2img':
            if data['model'].startswith("sdxl-"):
                with torch.no_grad():
                    outputs = current_model(
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_prompt_embeds=negative_conditioning,
                        negative_pooled_prompt_embeds=negative_pooled,
                        image=data['image_data'],
                        num_inference_steps=data['steps'],
                        width=data['width'],
                        height=data['height'],
                        guidance_scale=data['guidance'],
                        generator=data['seed'],
                        strength=data['strength'],
                        num_images_per_prompt=data['image_count'],
                    ).images
            else:
                with torch.no_grad():
                    outputs = current_model(
                        prompt_embeds=conditioning,
                        negative_prompt_embeds=negative_conditioning,
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
            if data['model'].startswith("sdxl-"):
                with torch.no_grad():
                    outputs = current_model(
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_prompt_embeds=negative_conditioning,
                        negative_pooled_prompt_embeds=negative_pooled,
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
            else:
                with torch.no_grad():
                    outputs = current_model(
                        prompt_embeds=conditioning,
                        negative_prompt_embeds=negative_conditioning,
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
        # elif model_type == 'controlnet_img2img' or model_type == 'openpose':
        #     with torch.no_grad():
        #         outputs = current_model(
        #             prompt_embeds=conditioning,
        #             negative_prompt_embeds=negative_conditioning,
        #             num_inference_steps=data['steps'],
        #             width=data['width'],
        #             height=data['height'],
        #             guidance_scale=data['guidance'],
        #             generator=data['seed'],
        #             strength=data['strength'],
        #             image=data['image_data'],
        #             num_images_per_prompt=data['image_count'],
        #         ).images
        elif model_type == 'txt2video':
            with torch.no_grad():
                outputs = current_model(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    width=data['width'],
                    height=data['height'],
                    num_inference_steps=data['steps'],
                    guidance_scale=data['guidance'],
                    generator=data['seed'],
                    num_frames=data['video_length'],
                )
                
        # return the model to the CPU if it was on the GPU:
    
        return outputs
                
            

    except Exception as e:

        if data['gpu_id'] == 1:
            current_model.to("cpu")
        if data['gpu_id'] == 0:
            current_model.to("cpu")

        error_message = str(e)
        error_message = error_message.replace("'", '"')

        log_error(f"Error processing request: {error_message}", data)

        # check if error message has anywhere in it "is not in list":
        error_aaaa = False
        if "is not in list" in error_message:
            error_aaaa = True
        if error_message == '"LayerNormKernelImpl" not implemented for "Half"' or error_aaaa:
            current_model.unload_lora_weights()
            current_model.unfuse_lora()
            if data['request_type'] == 'txt2img':
                txt2img_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'img2img':
                img2img_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'inpainting':
                inpainting_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'openpose':
                openpose_models[data['model']]['loaded'] = None
            error_message = error_message + " | Model Reloaded"
        # if the error has "Cannot generate a"
        if "Cannot generate a" in error_message:
            current_model.unload_lora_weights()
            current_model.unfuse_lora()
            if data['request_type'] == 'txt2img':
                txt2img_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'img2img':
                img2img_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'inpainting':
                inpainting_models[data['model']]['loaded'] = None
            elif data['request_type'] == 'openpose':
                openpose_models[data['model']]['loaded'] = None
            error_message = error_message + " | Model Reloaded"
            
        print("Error processing request:", error_message)
        log_error(f"Error processing request: {error_message}", data)
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

def add_metadata(image, metadata, data):
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
            "loras": data['lora'],
            "lora_strengths": data['lora_strengths'],
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
            meta = add_metadata(watermarked_image, metadata, data)
            buffered = io.BytesIO()
            watermarked_image.save(buffered, format="PNG", pnginfo=meta)
            img_str = base64.b64encode(buffered.getvalue()).decode()
        else:
            img_str = data['video_string']
            
        if data['save_image'] == True:
            # create dir if you need to:
            os.makedirs("./output_images", exist_ok=True)
            if model_type != "txt2video":
                # check if image already exists, if it does, add a number to the end of the filename:
                # if os.path.exists(f"./output_images/{request_id}.png"):
                #     i = 1
                #     while os.path.exists(f"./output_images/{request_id}_{i}.png"):
                #         i += 1
                #     watermarked_image.save(f"./output_images/{request_id}_{i}.png", pnginfo=meta)
                # else:
                watermarked_image.save(f"./output_images/{request_id}.png", pnginfo=meta)
            else:
                with open(f"./output_images/{request_id}.mp4", "wb") as file:
                    file.write(base64.b64decode(data['video_string']))
                    
        

        return {
            "width": output_image.width,
            "height": output_image.height,
            "base64": img_str
        }
            
    except Exception as e:
        print(f"Error saving image: {e}")
        log_error(f"Error saving image: {e}", data)
        return None












log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
        request_id = queue_item.request_id
        data = queue_item.data
        
        promptString = str(data['prompt'])
        
        # if prompt is a list then join it into a string:
        if isinstance(data['prompt'], list):
            promptString = ' '.join(data['prompt'])
        
        # data on multiple print lines for easier debugging
        # print("Request Type: " + str(data['request_type']) + " | Model: " + str(data['model']) + " Scheduler: " + str(data['scheduler']) + "\nSteps: " + str(data['steps']) + " | Width: " + str(data['width']) + "px | Height: " + str(data['height']) + "px\nSeed: " + str(data['seedNumber']) + " | Strength: " + str(data['strength']) + " | CFGuidance: " + str(data['guidance']) + " | Image Count: " + str(data['image_count'] )  + "\nPrompt: " + str(promptString) + "\nNegative Prompt: " + str(data['negative_prompt']) + "\nLora: " + str(data['lora']))

        printString = f"""
            Request Type: {data['request_type']} | Model: {data['model']} | Scheduler: {data['scheduler']} | Steps: {data['steps']} | Width: {data['width']}px | Height: {data['height']}px
            Seed: {data['seedNumber']} | Strength: {data['strength']} | CFGuidance: {data['guidance']} | Image Count: {data['image_count']} | Aspect Ratio: {data['aspect_ratio']} | Account ID: {data['accountId']}
        """
        
        print(printString)

        model_name = data['model']
        lora = data.get('lora', "NO")
        model_type = data['request_type']
        
                
        # if model_type is txt2img or img2img, get the model, else get the inpainting model:
        if model_type == 'txt2img':
            model = get_models.txt2img(model_name, data)
        elif model_type == 'img2img':
            model = get_models.img2img(model_name, data)
        elif model_type == 'inpainting':
            model = get_models.inpainting(model_name, data)
        elif model_type == 'txt2video':
            model = get_models.txt2video(model_name, data)
        # elif model_type == 'controlnet_img2img':
        #     model = get_controlnet_img2img_model(model_name, data)
        elif model_type == 'openpose':
            model = get_models.openpose(model_name, data)
        # elif model_type == 'latent_couple':
        #     model = get_txt2img_model(model_name, data)
        #     data['inpainting_original_option'] = True
        #     inpainting_model = get_inpainting_model(model_name, data)
        #     img2img_model = get_img2img_model(model_name, data)
            
        current_model = model
            
        if current_model is not None:

            if model_move_manual:
                if data['gpu_id'] == 1:
                    current_model.to("cuda:1")
                if data['gpu_id'] == 0:
                    current_model.to("cuda:0")

            current_model.unfuse_lora()
            current_model.unload_lora_weights()
            # print(model_type)

            if data['lora_strengths'] == [] or data['lora_strengths'] is None:
                data['lora_strengths'] = []
            
            if data['lora'] is not None and model_type != 'latent_couple':
                current_model = load_loras(request_id, current_model, lora, data)
                print("loras loaded successfully")
            if model_type != 'latent_couple':
                model_outputs = process_image(current_model, model_type, data, request_id)
                
            if model_move_manual:
                if data['gpu_id'] == 1:
                    current_model.to("cpu")
                if data['gpu_id'] == 0:
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
                        # img2img_model.unfuse_lora()
            #             img2img_model.unload_lora_weights()
                        
            if model_outputs is not None:
                image_data_list = []
                
                if model_type == 'txt2video':
                    model_outputs = model_outputs.frames[0]
                    
                # if model_type == 'txt2video':
                    
                #     img2img_model_video = get_img2img_model(model_name, data)
                    
                #     steps_before = data['steps']
                #     strength_before = data['strength']
                    
                #     data['steps'] = 30
                #     data['strength'] = 0.1
                    
                #     improved_frames = []
                    
                #     for img in model_outputs:
                #         data['image_data'] = img
                #         output = process_image(img2img_model_video, "img2img", data, request_id)
                #         improved_frames.append(output[0])
                        
                #     # img2img_model_video.to("cpu")
                        
                #     model_outputs = improved_frames
                        
                        
                #     data['steps'] = steps_before
                #     data['strength'] = strength_before
            
            
                
                if data.get('upscale', False) and model_type != 'txt2video':
                    
                    # make the directories if they don't exist:
                    if not os.path.exists(f"{data['gpu_id']}-toupscale"):
                        os.makedirs(f"{data['gpu_id']}-toupscale")
                        
                    if not os.path.exists(f"{data['gpu_id']}-upscaled"):
                        os.makedirs(f"{data['gpu_id']}-upscaled")
                        
                    # remove the toupscale images:
                    for files in os.listdir(f"{data['gpu_id']}-toupscale"):
                        os.remove(f"{data['gpu_id']}-toupscale/{files}")
            
                    # save image as "og_image.png"
                    for index, img in enumerate(model_outputs):
                        img.save(f"{data['gpu_id']}-toupscale/og-image-{index}.png")
                
                    import subprocess

                    # Define the command to run as a list of arguments
                    command = ["./esrganvulkan/realesrgan-ncnn-vulkan.exe", "-n", "realesrgan-x4plus-anime", "-i", f"{data['gpu_id']}-toupscale", "-o", f"{data['gpu_id']}-upscaled", "-f", "png", "-s", "4", "-t", "256"]
                    # python Real-ESRGAN/inference_realesrgan.py -n realesrgan-x4plus-anime -i og_image.png -o upscaled_image.png -f png

                    # Run the command and wait for it to complete, capturing the output
                    result = subprocess.run(command, capture_output=True, text=True)

                    # create array with the upscaled images:
                    upscaled_images = []
                    
                    # load the upscaled images into memory:
                    for index, img in enumerate(model_outputs):
                        img = Image.open(f"{data['gpu_id']}-upscaled/og-image-{index}.png")
                        upscaled_images.append(img)
                        
                    model_outputs = upscaled_images
                    
                    
                if model_type == 'txt2video':
                    processed_frames = []
                    for frame in model_outputs:
                        frame = add_watermark(frame, "JSCammie.com", data)
                        processed_frames.append(frame)
                        
                    data['video_string'] = export_to_mp4(processed_frames, "animation.mp4")
                    
                    model_outputs = [model_outputs[0]]
                
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
                
                saving_object = {
                    "data": data,
                    "images": PIL_Images,
                    "status": "pending"
                }                    
                    
                print("Time to save images: " + str(time.time() - timeBeforeSave))
                
                
            time_added_to_queue = data['queued_time']
            total_time = time.time() - time_added_to_queue
            
            average_queue_times.append(total_time)

            results[request_id] = {
                "images": image_data_list,
                "additionalInfo": {
                    "seed": data['seedNumber'],
                    "executiontime": time.time() - start_time,
                    "totaltime": total_time,
                    "timestamp": time.time()
                }
            }
            
            queue_item.status = "completed"
            return "processed"
            
        else:
            error_message = "Invalid model name"
            print("Error processing request:", error_message)
            results[request_id] = {"status": "error", "message": error_message}

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        results[request_id] = {"status": "error", "message": error_message}




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
                    # print("Finished processing")
                    hash_queue_busy = False
                    hash_queue.remove(queue_item)

        # Sleep if no unprocessed request is found
        if not any(item['status'] == "pending" for item in hash_queue):
            await asyncio.sleep(0.5)  # Corrected to use await
            
def process_queue_0():
    global processor_busy_0

    
    while True:
        time.sleep(0.01)  # Short sleep to prevent CPU overutilization
        if not processor_busy_0:
            
            if len(results) > 0:
                # remove any items from the results that have a timestamp that is over 10 minutes old (600 seconds):
                current_time = time.time()
                print("Current Time: " + str(current_time))
                for key in list(results.keys()):
                    if results[key].get('additionalInfo', {}).get('timestamp', 0) < current_time - 600:
                        del results[key]
                        
            if request_queue_0:  # Check if the queue is not empty
                queue_item = request_queue_0[0]  # Get the first item
                if queue_item.status == "queued":
                    queue_item.status = "waiting"
                    processor_busy_0 = True
                    
                    result = process_request(queue_item)
                    
                    # print results count:
                    print("0 | Results Count: " + str(len(results)))
                    print("0 | Queue Count: " + str(len(request_queue_0)))
                    if len(average_queue_times) > 100:
                        average_queue_times.pop(0)
                    if len(average_queue_times) > 0:
                        print(f"Average Queue Time: {sum(average_queue_times) / len(average_queue_times)}")
                    if result == "processed":
                        processor_busy_0 = False
                        request_queue_0.remove(queue_item)  # Remove the item from the queue
                    elif queue_item.status in ["completed", "error", "skipped"]:
                        processor_busy_0 = False
                        request_queue_0.remove(queue_item)  # Remove the item from the queue
            
        # Sleep if no unprocessed request is found
        if not any(item.status == "queued" for item in request_queue_0):
            time.sleep(0.5)


def process_queue_1():
    global processor_busy_1
    
    while True:
        time.sleep(0.01)  # Short sleep to prevent CPU overutilization
        if not processor_busy_1:
            
            if request_queue_1:  # Check if the queue is not empty
                queue_item = request_queue_1[0]  # Get the first item
                if queue_item.status == "queued":
                    queue_item.status = "waiting"
                    processor_busy_1 = True

                    result = process_request(queue_item)
                    
                    # print results count:
                    print("1 | Results Count: " + str(len(results)))
                    print("1 | Queue Count: " + str(len(request_queue_1)))
                    if len(average_queue_times) > 100:
                        average_queue_times.pop(0)
                    if len(average_queue_times) > 0:
                        print(f"Average Queue Time: {sum(average_queue_times) / len(average_queue_times)}")
                    if result == "processed":
                        processor_busy_1 = False
                        request_queue_1.remove(queue_item)  # Remove the item from the queue
                    elif queue_item.status in ["completed", "error", "skipped"]:
                        processor_busy_1 = False
                        request_queue_1.remove(queue_item)  # Remove the item from the queue
            
        # Sleep if no unprocessed request is found
        if not any(item.status == "queued" for item in request_queue_1):
            time.sleep(0.5)

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
        # print("Fast pass is nonexistent or the same as the accountId.")
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
    # print(f"Checking fast pass: {userid} against the fast pass map: {fast_passes_map}")
    
    # Ensure thread-safe read
    with threading.Lock():
        # check if the userid is in the banned_users_map, there are 4 categories of banned users:
        
        if banned_users_map[request_type].get(userid, None) is not None:
            return banned_users_map[request_type][userid]
        else:
            return False

@app.route('/cancel_request/<request_id>', methods=['GET'])
def cancel_request(request_id):
    try:
        for item in request_queue_0:
            if item.request_id == request_id:
                # check the position of the item in the queue, if its 1st or 2nd then return processing:
                index = request_queue_0.index(item)
                if index < 2:
                    return jsonify({"status": "processing", "message": "Request is currently being processed"}), 200
                item.status = "cancelled"
                request_queue_0.remove(item)
                return jsonify({"status": "cancelled", "message": "Cancelled Request"}), 200
            else:
                for item in request_queue_1:
                    if item.request_id == request_id:
                        index = request_queue_0.index(item)
                        if index < 2:
                            return jsonify({"status": "processing", "message": "Request is currently being processed"}), 200
                        item.status = "cancelled"
                        request_queue_1.remove(item)
                        return jsonify({"status": "cancelled", "message": "Cancelled Request"}), 200
                    
        return jsonify({"status": "not found", "message": "Invalid request_id, Not Found"}), 404
    
    except Exception as e:
        return generate_error_response(str(e), 500)

@app.route('/queue_position/<request_id>', methods=['GET'])
def check_queue_position(request_id):
    # Loop through the queues and find the position for the given request_id
    # for index, item in enumerate(request_queue):
    #     if item.request_id == request_id:
    #         return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": len(request_queue)}), 200
    # if request_id in results:
    #     if results[request_id].get("status") == "error":
    #         return jsonify({"status": "error", "message": results[request_id].get("message")}), 200
    #     return jsonify({"status": "completed", "request_id": request_id}), 200
    # return jsonify({"status": "not found", "message": "Invalid request_id"}), 404
    
    for index, item in enumerate(request_queue_0):
        if item.request_id == request_id:
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": len(request_queue_0)}), 200
    for index, item in enumerate(request_queue_1):
        if item.request_id == request_id:
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": len(request_queue_1)}), 200
    if request_id in results:
        if results[request_id].get("status") == "error":
            return jsonify({"status": "error", "message": results[request_id].get("message")}), 200
        return jsonify({"status": "completed", "request_id": request_id}), 200
    
    return jsonify({"status": "not found", "message": "Invalid request_id"}), 404

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    result = results.get(request_id)
    # remove the result from the results dictionary:
    results.pop(request_id, None)
    if result:
        return jsonify(result)
    else:
        return jsonify({"status": "processing"}), 202 
    
import open_clip

# load tokenizer openclip:
# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

print("Tokenizer loaded")

@app.route('/token-length', methods=['POST'])
def token_length():
    data = request.json
    prompt = data.get('prompt', "")
    negative_prompt = data.get('negativeprompt', "")
    
    # get the tokens for the prompt and then the negative prompt:
    prompt_tokens = tokenizer.encode(prompt)
    negative_prompt_tokens = tokenizer.encode(negative_prompt)
    
    # print(f"Prompt: {len(prompt_tokens)}, Negative Prompt: {len(negative_prompt_tokens)}")
    
    return jsonify({"prompt": len(prompt_tokens), "negative_prompt": len(negative_prompt_tokens)})

# Define a helper function to check JSON serializability
def serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

@app.route('/get-all-queue', methods=['GET'])
def get_all_queue():

    queue_0 = []
    queue_1 = []

    # Process each item in the queue, only keeping serializable parts
    for item in request_queue_0:
        item_dict = {k: v for k, v in item.__dict__.items() if serializable(v)}
        queue_0.append(item_dict)

    for item in request_queue_1:
        item_dict = {k: v for k, v in item.__dict__.items() if serializable(v)}
        queue_1.append(item_dict)

    return jsonify({"queue_0": queue_0, "queue_1": queue_1})
                
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        if global_settings.get('maintenance', False):
            return generate_error_response("Maintenance Mode is currently enabled, the requests that are already in the queue are being completed, please wait a minute or two and try again.", 503)

        data = request.json
        
        if global_settings.get('sdxl', False):
            if data['model'].startswith("sdxl-"):
                return generate_error_response("SDXL is currently disabled, please use the other models instead.", 503)
            
        if data['request_type'] == "txt2video":
            return generate_error_response("Text to video is currently disabled.", 503)
        
        data['gpu_id'] = 0
                    
        # if data['model'].startswith("sdxl-"):
        #     data['gpu_id'] = 0
        # else:
        #     data['gpu_id'] = 1
            
        # cap the steps to global_settings['sdxl_max_steps']:
        if data['model'].startswith("sdxl-"):
            if data['steps'] > global_settings['sdxl_max_steps']:
                data['steps'] = global_settings['sdxl_max_steps']
                
        if data.get('aspect_ratio', None) is not None:
            if data['aspect_ratio'] == "portrait":
                if data['model'].startswith("sdxl-"):
                    data['width'] = 768
                    data['height'] = 1024
                else:
                    data['width'] = 512
                    data['height'] = 768
                    
                
            elif data['aspect_ratio'] == "landscape":
                if data['model'].startswith("sdxl-"):
                    data['width'] = 1024
                    data['height'] = 768
                else:
                    data['width'] = 768
                    data['height'] = 512
                    
                
            elif data['aspect_ratio'] == "square":
                if data['model'].startswith("sdxl-"):
                    data['width'] = 1024
                    data['height'] = 1024
                else:
                    data['width'] = 512
                    data['height'] = 512
                
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
                
                
                
            else:
                data['width'] = data['width'] * global_settings['sd15_resolution_multiplier']
                data['height'] = data['height'] * global_settings['sd15_resolution_multiplier']
   
                
        if data['model'].startswith("sdxl-"):
            if len(data['lora']) > global_settings['max_loras_sdxl']:
                return generate_error_response(f"Maximum number of Lora options is {global_settings['max_loras_sdxl']}", 400)
        else:
            if len(data['lora']) > global_settings['max_loras']:
                return generate_error_response(f"Maximum number of Lora options is {global_settings['max_loras']}", 400)
            
        # if data['request_type'] == "inpainting": and the data['model'] doesnt start with sdxl- return error:
        # if data['request_type'] == "inpainting" or data['request_type'] == "img2img":
        #     if data['request_type'] == "inpainting":
        #         return generate_error_response("Inpainting is currently disabled.", 503)
        #     if data['request_type'] == "img2img":
        #         return generate_error_response("Image to image is currently disabled.", 503)
                 
        
        data['width'] = round_to_multiple_of_eight(data['width'])
        data['height'] = round_to_multiple_of_eight(data['height'])
        
        import process_input_data

        # Validate and preprocess the input data
        validated_data, error_message = process_input_data.validate_input_data(data)
        if error_message:
            return generate_error_response(error_message, 400)
        
        
        data['width'] = round_to_multiple_of_eight(data['width'])
        data['height'] = round_to_multiple_of_eight(data['height'])
        

        # Check for banned users and fast pass
        account_id = validated_data['accountId']
        request_type = validated_data['request_type']
        fastpass = validated_data.get('fastpass', None)
        
        if request_type == "latent_couple":
            return generate_error_response("Latent couple is currently disabled.", 503)
        
        if request_type == "openpose":
            return generate_error_response("Openpose is currently disabled.", 503)

        if account_id is not None:
            ban_check_result = check_banned_users(account_id, request_type)
            if ban_check_result is not False:
                return generate_error_response(f"User {account_id} is banned for {ban_check_result['reason']}, in the {request_type} category", 400)
            
        if int(account_id) != 0 and int(account_id) != 1194813360529735711:
            # if account_id is already in the request_queue_0 or request_queue_1, return an error:
            if data['gpu_id'] == 0:
                for item in request_queue_0:
                    if int(item.data['accountId']) == int(account_id):
                        index = request_queue_0.index(item)
                        if index > 2:
                            # set the generate_retries to 0 if it doesn't exist:
                            if 'generate_retries' not in item.data:
                                item.data['generate_retries'] = 0

                            print(f"Generate Retries: {item.data['generate_retries']}")
                            if item.data['generate_retries'] == 3 or item.data['generate_retries'] > 3:
                                # cancel the request:
                                item.status = "cancelled"
                                request_queue_0.remove(item)
                                return generate_error_response(f"User {account_id} already has a request in the queue, the request has been cancelled", 400)
                            else:
                                item.data['generate_retries'] += 1
                                return generate_error_response(f"User {account_id} already has a request in the queue, retry {3 - item.data['generate_retries']} more times to cancel the request", 400)
                        else:
                            return generate_error_response(f"User {account_id} already has a request in the queue", 400)
            elif data['gpu_id'] == 1:
                for item in request_queue_1:
                    if int(item.data['accountId']) == int(account_id):
                        index = request_queue_0.index(item)
                        if index > 2:
                            # set the generate_retries to 0 if it doesn't exist:
                            if 'generate_retries' not in item.data:
                                item.data['generate_retries'] = 0

                            if item.data['generate_retries'] == 3:
                                # cancel the request:
                                item.status = "cancelled"
                                request_queue_1.remove(item)
                                return generate_error_response(f"User {account_id} already has a request in the queue, the request has been cancelled", 400)
                            else:
                                item.data['generate_retries'] += 1
                                return generate_error_response(f"User {account_id} already has a request in the queue, retry {3 - item.data['generate_retries']} more times to cancel the request", 400)
                        else:
                            return generate_error_response(f"User {account_id} already has a request in the queue", 400) 
                        
        # check if the ip is already in the queue and cancel after 3 retries:
        if data['ip'] is not None:
            if data['ip'] != "123123123":
                for item in request_queue_0:
                    if item.data['ip'] == data['ip']:
                        index = request_queue_0.index(item)
                        if index > 2:
                            # set the generate_retries to 0 if it doesn't exist:
                            if 'generate_retries' not in item.data:
                                item.data['generate_retries'] = 0

                            if item.data['generate_retries'] == 3:
                                # cancel the request:
                                item.status = "cancelled"
                                request_queue_0.remove(item)
                                return generate_error_response(f"IP {data['ip']} already has a request in the queue, the request has been cancelled", 400)
                            else:
                                item.data['generate_retries'] += 1
                                return generate_error_response(f"IP {data['ip']} already has a request in the queue, retry {3 - item.data['generate_retries']} more times to cancel the request", 400)
                        else:
                            return generate_error_response(f"IP {data['ip']} already has a request in the queue", 400)
                for item in request_queue_1:
                    if item.data['ip'] == data['ip']:
                        index = request_queue_1.index(item)
                        if index > 2:
                            # set the generate_retries to 0 if it doesn't exist:
                            if 'generate_retries' not in item.data:
                                item.data['generate_retries'] = 0

                            if item.data['generate_retries'] == 3:
                                # cancel the request:
                                item.status = "cancelled"
                                request_queue_1.remove(item)
                                return generate_error_response(f"IP {data['ip']} already has a request in the queue, the request has been cancelled", 400)
                            else:
                                item.data['generate_retries'] += 1
                                return generate_error_response(f"IP {data['ip']} already has a request in the queue, retry {3 - item.data['generate_retries']} more times to cancel the request", 400)
                        else:
                            return generate_error_response(f"IP {data['ip']} already has a request in the queue", 400)
        
        # if the ip is already in the queue, return an error:
        if data['ip'] is not None:
            if data['ip'] != "123123123":
                for item in request_queue_0:
                    if item.data['ip'] == data['ip']:
                        print("IP is in the queue")
                        print(f"IP: {data['ip']}, Request IP: {item.data['ip']}")
                        return generate_error_response("You are already in the queue", 400)
                for item in request_queue_1:
                    if item.data['ip'] == data['ip']:
                        print("IP is in the queue")
                        print(f"IP: {data['ip']}, Request IP: {item.data['ip']}")
                        return generate_error_response("You are already in the queue", 400)
        
            

        fastpass_enabled, error_message = check_fast_pass(fastpass, validated_data)
        if error_message:
            return generate_error_response(error_message, 400)
        
        if global_settings.get('upscale', False):
            if validated_data['upscale']:
                # if validated_data['model'].startswith("sdxl-"):
                #     return generate_error_response("Upscaling is currently disabled for SDXL models!", 503)
                return generate_error_response("Upscaling is currently disabled.", 503)
            
        if validated_data['model'].startswith("sdxl-"):
            if validated_data['upscale']:
                return generate_error_response("Upscaling is currently disabled for SDXL models!", 503)
                
        # Check if the model is valid
        model_name = validated_data['model']
        if model_name not in txt2img_models:
            return generate_error_response("Invalid model name", 400)

        # Prepare the data for the request
        data = validated_data

        request_id = str(uuid.uuid4())
        data['request_id'] = request_id
        queue_item = QueueRequest(request_id, data)

        # Check for duplicate requests
        if data['gpu_id'] == 0:
            if queue_item.data in [item.data for item in request_queue_0]:
                return generate_error_response("Duplicate request", 400)
        elif data['gpu_id'] == 1:
            if queue_item.data in [item.data for item in request_queue_1]:
                return generate_error_response("Duplicate request", 400)

        
        
        # print(f"Fast pass: {fastpass}, Fast pass enabled: {fastpass_enabled}")

        if data['gpu_id'] == 0:
            if fastpass and fastpass_enabled:
                request_queue_0.insert(0, queue_item)
            else:
                request_queue_0.append(queue_item)
                
            position = len(request_queue_0)  # Current position in the queue is its length
        elif data['gpu_id'] == 1:
            # data['gpu_id'] = 0
            if fastpass and fastpass_enabled:
                request_queue_1.insert(0, queue_item)
            else:
                request_queue_1.append(queue_item)
                
            position = len(request_queue_1)  # Current position in the queue is its length

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
    "model": "furryblend",
    "prompt": "((Masterpiece)), high quality, studio quality, intricate details, 4k, solo, (emphasis lines), 2d, Coco Bandicoot, having sex, riding, facing viewer, large breasts, thick thighs, ahegao, saliva, penis stomach bulge, cum",
    "negativeprompt": "3d, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "steps": 20,
    "aspect_ratio": "portrait",
    "seed": 123123123,
    "quantity": 4,
    "request_type": "txt2img",
    "lora": ['character-cocobandicootcrashbandicoot'],
    "upscale": False,
    "save_image": True
}

generateTestJson1video = {
    "model": "furryblend",
    "prompt": "((Masterpiece)), high quality, studio quality, intricate details, 4k, solo, (emphasis lines), 2d, Coco Bandicoot, facing viewer, wearing skirt",
    "negativeprompt": "3d, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "steps": 20,
    "aspect_ratio": "portrait",
    "seed": 123123123,
    "quantity": 1,
    "request_type": "txt2video",
    "lora": ['character-cocobandicootcrashbandicoot'],
    "upscale": True,
    "video_length": 16
}


generateTestJson1videoSDXL = {
    "model": "sdxl-ponydiffusion",
    "prompt": "score_9, score_8_up, 2d, coco bandicoot, large breasts, thick thighs, denim shorts, black crop top, in city",
    "negativeprompt": "(score_6, score_5, 3d, hyperrealistic, octane renderer, monochrome, black and white, rough sketch)",
    "steps": 20,
    "aspect_ratio": "portrait",
    "seed": 123123123,
    "quantity": 1,
    "request_type": "txt2video",
    "lora": [],
    "upscale": True,
    "video_length": 16
}


generateTestJsonSDXL = {
    "model": "sdxl-ponydiffusion",
    "prompt": "(score_9, score_8_up, 2d), coco bandicoot, denim shorts, black crop top, in city",
    "negativeprompt": "(score_6, score_5, 3d, hyperrealistic, monochrome, black and white, rough sketch)",
    "steps": 20,
    "aspect_ratio": "portrait",
    "seed": 123123123,
    "quantity": 4,
    "request_type": "txt2img",
    "lora": [],
    "upscale": False,
    "save_image": True
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
        
def test_token_length(test_data):
    with app.test_client() as client:
        response = client.post('/token-length', json=test_data)
        status_code = response.status_code
        # Additional assertions or checks on the response can be added here
        print(f"Status Code: {status_code}")
        print(response.json)
        
token_length_test_data = {
    "prompt": "1girl, amy rose, nude, sexy",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective"
}
        
# test_token_length(token_length_test_data)

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

# def test_all_loras():
    
#     for lora_name, lora_data in lora_weights_map.items():
#         if lora_name == "background":
#             for lora in lora_data:
#                 generateTest = {
#                     "model": "fluffysonic",
#                     "prompt": f"1girl, amy rose, nude, sexy",
#                     "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective,",
#                     "steps": 20,
#                     "width": 512,
#                     "height": 512,
#                     "seed": 24682468,
#                     "accountId": 1039574722163249233,
#                     "quantity": 1,
#                     "request_type": "txt2img",
#                     "lora": [f"{lora}"],
#                     "upscale": False,
#                     "loraTest": True
#                 }
#                 test_generate_image(generateTest)
                
# test_all_loras()


test_video_1 = {
    "model": "fluffysonic",
    "prompt": "1girl, amy rose, nude, sexy",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "steps": 25,
    "aspect_ratio": "square",
    "guidance": 7,
    "seed": 123123123,
    "quantity": 1,
    "request_type": "txt2video",
    "lora": [],
    "upscale": True,
    "video_length": 16
}



generateTestJsonImg2Img = {
    "model": "fluffysonic",
    "prompt": "1girl, amy rose, nude, sexy",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "steps": 25,
    "aspect_ratio": "portrait",
    "guidance": 7,
    "seed": 123123123,
    "quantity": 1,
    "request_type": "img2img",
    "lora": [],
    "input_image": "input.png",
    "strength": 0.7,
    "save_image": True,
}

generateTestJsontxt2imgLandscape = {
    "model": "fluffysonic",
    "prompt": "1girl, amy rose, nude, sexy",
    "negativeprompt": "worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "steps": 25,
    "aspect_ratio": "landscape",
    "guidance": 7,
    "seed": 123123123,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": [],
    "upscale": False,
    "save_image": True,
    "ip": "123123123"
}
    
    
    
# test_generate_image(generateTestJsonLatentCouple)
# test_generate_image(generateTestJsonSDXL)
# test_generate_image(generateTestJson1)
# test_generate_image(test_video_1)
# test_generate_image(generateTestJson1videoSDXL)
# test_generate_image(generateTestJson2)
# test_generate_image(generateTestJson2a)
# test_generate_image(generateTestJson22)
# test_generate_image(generateTestJson222)
# test_generate_image(generateTestJson3)
# test_generate_image(generateTestJsonImg2Img)
test_generate_image(generateTestJsontxt2imgLandscape)




    
def run_flask_app():
    app.run(host='0.0.0.0', port=5003)

def start_background_tasks():
    threading.Thread(target=process_queue_0, daemon=True).start()
    threading.Thread(target=process_queue_1, daemon=True).start()
    threading.Thread(target=run_flask_app, daemon=True).start()

async def main():
    start_background_tasks()
    await process_hash_queue()

if __name__ == '__main__':
    # if model_move_manual == False:
        
        # for model_name, model_info in txt2img_models.items():
        #     if model_name.startswith("sdxl-"):
        #         data = {'gpu_id': 0}
        #     else:
        #         data = {'gpu_id': 0}
        #     if model_info['loaded'] is None:
        #         model_info['loaded'] = load_models.txt2img(model_name, data, model_info['model_path'])
        #         print(f"Loaded {model_name}, txt2img")
                
        # for model_name, model_info in img2img_models.items():
        #     if model_name.startswith("sdxl-"):
        #         data = {'gpu_id': 0}
        #     else:
        #         data = {'gpu_id': 0}
        #     if model_info['loaded'] is None:
        #         model_info['loaded'] = load_models.img2img(model_name, data, model_info['model_path'])
        #         print(f"Loaded {model_name}, img2img")
                
        # for model_name, model_info in inpainting_models.items():
        #     if model_name.startswith("sdxl-"):
        #         data = {'gpu_id': 0}
        #     else:
        #         data = {'gpu_id': 0}
        #     if model_info['loaded'] is None:
        #         model_info['loaded'] = load_models.inpainting(model_name, data, model_info['model_path'])
        #         print(f"Loaded {model_name}, inpainting")
    asyncio.run(main())
    print(torch.__version__)
    print("Startup time: " + str(time.time() - program_start_time) + " seconds")
    app.run(host='0.0.0.0', port=5003)