import base64
from flask import Flask, request, jsonify, send_file
import logging
from flask_cors import CORS
import io
import random
import json
import threading
import yaml
import uuid
import torch
import tomesd
from diffusers import ControlNetModel, StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline
from diffusers.utils import load_image, export_to_video
from diffusers.models.attention_processor import AttnProcessor2_0
import time
from io import BytesIO
# from transformers import AutoTokenizer, AutoModelForCausalLM

from concurrent.futures import ThreadPoolExecutor

import cv2

from controlnet_aux import OpenposeDetector

import numpy as np

import os
import datetime

program_start_time = time.time()

# Configuration Loading
with open('config.json', 'r') as f:
    config = json.load(f)

# Model Loading
furry_model_path = config["furry_model_path"]
sonic_model_path = config["sonic_model_path"]
aing_model_path = config["aing_model_path"]
flat2DAnimerge_model_path = config["flat2DAnimerge_model_path"]
realisticVision_model_path = config["realisticVision_model_path"]

try:
        
    img2img_models = {}
    
    inpainting_models = {
        'inpainting': {'loaded':None, 'model_path': './models/inpainting/SonicDiffusionV4-inpainting.inpainting.safetensors', 'scheduler': EulerAncestralDiscreteScheduler},
        'furry': {'loaded':None, 'model_path': furry_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }

    txt2img_models = {
        'furry': {'loaded':None, 'model_path': furry_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }
    
    controlnet_img2img_models = {
        'furry': {'loaded':None, 'model_path': furry_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }
    
    openpose_models = {
        'furry': {'loaded':None, 'model_path': furry_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'sonic': {'loaded':None, 'model_path': sonic_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'aing': {'loaded':None, 'model_path': aing_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'flat2DAnimerge': {'loaded':None, 'model_path': flat2DAnimerge_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
        'realisticVision': {'loaded':None, 'model_path': realisticVision_model_path, 'scheduler': EulerAncestralDiscreteScheduler},
    }
    
    # for each model in txt2img_models that doesnt have a save_pretrained folder, create one by using StableDiffusionPipeline, loading the model and using the name as the final folder:
    for model_name, model_info in txt2img_models.items():
        if not os.path.exists('./models/' + model_name):
            print("Creating folder for " + model_name)
            try:
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
        
    generator = torch.Generator(device="cuda")
        
    torch.backends.cuda.matmul.allow_tf32 = True

except Exception as e:
    print(f"Failed to load the model: {e}")
    raise




def create_and_load_inpainting_model(model_path, name, scheduler, model_type, data):
    if name == "inpainting":
        print("\nLoading Inpainting model")
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
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
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
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
    pipeline.load_textual_inversion("./embeddings/EasyNegativeV2.safetensors")
    pipeline.load_textual_inversion("./embeddings/BadDream.pt")
    pipeline.load_textual_inversion("./embeddings/boring_e621_v4.pt")
    pipeline.enable_model_cpu_offload()
    
    return pipeline


def create_and_load_model(model_path, name, scheduler, model_type, data):
    tomes_ratio=0.1

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
    tomesd.apply_patch(pipeline, ratio=tomes_ratio)
    pipeline.enable_vae_tiling()
    pipeline.enable_model_cpu_offload()
    
    components = pipeline.components
    components['safety_checker'] = None
    
    imgpipeline = StableDiffusionImg2ImgPipeline(**components, requires_safety_checker=False)

    img2img_models[name] = imgpipeline

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
    pipeline.enable_model_cpu_offload()
    
    components = pipeline.components
    components['safety_checker'] = None

    return pipeline
    
    
    
    
def get_txt2img_model(name, data):
    model_info = txt2img_models[name]
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    return model_info['loaded']

def get_inpainting_model(name, data):
    if data['inpainting_original_option'] == False:
        name = "inpainting"
    model_info = inpainting_models[name]
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_inpainting_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    return model_info['loaded']

def get_controlnet_img2img_model(name, data):
    model_info = controlnet_img2img_models[name]
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_controlnet_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
    return model_info['loaded']

def get_openpose_model(name, data):
    model_info = openpose_models[name]
    if model_info['loaded'] is None:
        model_info['loaded'] = create_and_load_controlnet_model(model_info['model_path'], name, model_info['scheduler'], data['request_type'], data)
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
        self.status = "waiting"

request_queue = []  # Use a list instead of the queue module for more control
# Dictionary to hold results indexed by request_id
results = {}
current_position = 0

def contains_any(string, substring_list):
    """Check if 'string' contains any of the substrings in 'substring_list'."""
    return any(sub in string for sub in substring_list)

def load_loras(request_id, current_model, lora_items):
    for item in lora_items:
                                    
        category, key = item.split('-', 1)
        lora_category = lora_weights_map.get(category, {})
        lora_data = lora_category.get(f"{category}-{key}")  # Ensure to use the full key

        try:
            if lora_data:
                strength = 1.0  
                # Check if 'strength' key exists in lora_data
                if 'strength' in lora_data:
                    strength = lora_data['strength']
                    
                if category == "character":
                    strength = 0.5
                    
                print(f"Found data for {item}: {lora_data['name']} - strength: {strength}")
                current_model.load_lora_weights(lora_data['lora'], low_cpu_mem_usage=False, ignore_mismatched_sizes=True, cross_attention_kwargs={"scale": strength} )
            else:
                print(f"No data found for {item}")
        
        except Exception as e:
            results[request_id] = {"error": str(e)}
            print("Error processing request:", e)

def process_image(current_model, model_type, data, request_id, save_image=False):
    try:
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
                if save_image == False:
                    return outputs
                else:
                    outputs[0].save("txt2img.png")
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
                    clip_skip=1,
                ).images
                if save_image == False:
                    return outputs
                else:
                    outputs[0].save("img2img.png")
        elif model_type == 'inpainting':
            print("Inpainting Generation Process Started")
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
                    clip_skip=1,
                    ).images
                if save_image == False:
                    return outputs
                else:
                    outputs[0].save("inpainting.png")
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
                    clip_skin=1,
                ).images
                return outputs

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        results[request_id] = {"status": "error", "message": error_message}
        return "CONTINUE"
    
# model_instance = get_txt2img_model('furry')
# process_image(model_instance, 'txt2img', {'prompt': '1girl, amy rose, glossy skin, shiny skin, (masterpiece, soft lighting, studio lighting, high quality, high detail, detailed background), in city, neon, glowing, rainging, bright, cute, fluffy, furry, wearing thigh highs, wearing croptop, looking at viewer, tan belly, bloom, bokeh, lens flare, sunlight, rainbow, crowded street path, street light, ', 'negative_prompt': '', 'image_count': 1, 'steps': 20, 'width': 512, 'height': 512, 'guidance': 6, 'seed': generator.manual_seed(69420) }, 'test', save_image=True)
        

def add_metadata(image, metadata):
    meta = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        meta.add_text(str(key), str(value))
    return meta

def save_image(request_id, output_image, model_type, data, image_index=0):
    try:
        
        if data['upscale'] == True:
            font_size = 48
        else:
            font_size = 24

        
        # Create the directory structure
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        no_no_list = ['child', 'kid', 'teen', 'teenager', 'loli', 'children', 'preteen', 'toddlercon', 'toddler', 'lolicon', 'oppai_loli', 'daughter','son']

        # create accountId_string based off the data['accountId'] :
        accountId_string = ""
        
        if data['accountId'] is not None:
            accountId_string = f"{data['accountId']}"
        else:
            accountId_string = "None"
        

        
        folder_path = f"./generations/{model_type}/{accountId_string}/"
        if contains_any(data['prompt'], no_no_list):
            folder_path = f"./generations/{model_type}/{accountId_string}/c/"
        folder_path += date_str
        os.makedirs(folder_path, exist_ok=True)
        
        if data['upscale'] is not False:
            
            upscale_model = StableDiffusionLatentUpscalePipeline.from_pretrained(
                'stabilityai/sd-x2-latent-upscaler',
                torch_dtype=torch.float16
            )
            upscale_model.unet.set_attn_processor(AttnProcessor2_0())
            upscale_model.enable_model_cpu_offload()
            upscale_model.enable_attention_slicing()
            # upscale_model = upscale_model.to("cuda")
            
            if isinstance(output_image, Image.Image):
                print("The image is a PIL.Image.Image type.")
            elif isinstance(output_image, np.ndarray):
                print("The image is a numpy.ndarray type.")
            elif isinstance(output_image, torch.Tensor):
                print("The image is a torch.Tensor type.")
            else:
                print(f"Unknown image type: {type(output_image)}")
                
            # convert the image to a torch tensor:
            output_image_open = Image.open(BytesIO(output_image))

            
            with torch.inference_mode():
                output_image = upscale_model(
                    prompt = data['prompt'],
                    image = output_image_open,
                    generator=data['seed'],
                    num_inference_steps=10,
                ).outputs[0]
            
            # clear torch memory:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()    

        # Add watermark
        watermarked_image = add_watermark(output_image, "NeverSFW.gg", font_size)
        # Add metadata
        metadata = {
            "request_id": request_id,
            "model_type": model_type,
            "prompt": data['prompt'],
            "loras": data['lora'],
            "steps": data['steps'],
            "CFG": data['guidance'],
            "upscaled": data['upscale'],
            "generation_date": datetime.datetime.now().isoformat(),
            "accountId": data['accountId']
        }
        meta = add_metadata(watermarked_image, metadata)

         # Generate a unique filename for each image
        image_filename = f"{request_id}_{image_index}.png"
        full_path = os.path.join(folder_path, image_filename)

        # Save the watermarked and metadata-added image
        # if data['accountId'] is not None:
        #     if data['accountId'] == "0":
        #         watermarked_image.save(full_path, format="PNG", pnginfo=meta)
        #         print(data['accountId'] + "AccountId, Image saved to: " + full_path)
                
        watermarked_image.save(full_path, format="PNG", pnginfo=meta)

        # Convert image to Base64 for JSON object
        buffered = io.BytesIO()
        watermarked_image.save(buffered, format="PNG", pnginfo=meta)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Create JSON object with image details
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
    return round(number / 8) * 8

def process_requests():
    """
    Process requests from the request queue.

    This function continuously checks the request queue for new requests and processes them accordingly.
    It retrieves the necessary data from each request, performs the required operations, and stores the results.

    Note:
    - The request queue is assumed to be a global variable.
    - The results dictionary is assumed to be a global variable.

    Returns:
    None
    """
    print("Started processing thread...")
    while True:
        if request_queue:
            try:
                start_time = time.time()
                nextReq = (" | Queue Length: " + str(len(request_queue)))
                print("\nProcessing next request..." + nextReq)
                queue_item = request_queue[0]
                request_id = queue_item.request_id
                formdata = queue_item.data
                
                negativePromptFinal = "EasyNegativeV2, boring_e621_v4, " + formdata.get("negativeprompt", "")
                
                
                data = {
                    "model": formdata.get('model'),
                    "prompt": formdata.get('prompt'),
                    "negative_prompt": negativePromptFinal,
                    "image_count": int(formdata.get("quantity")),
                    "steps": int(formdata.get("steps", 20)),
                    "width": int(formdata.get("width", 512)),
                    "height": int(formdata.get("height", 512)),
                    "seed": int(formdata.get("seed", -1)),
                    "strength": float(formdata.get("strength", 0.75)),
                    "guidance": float(formdata.get("guidance", 5)),
                    "image_data": formdata.get("image_data", None),
                    "mask_data": formdata.get("mask_data", None),
                    "lora": formdata.get('lora', None),
                    "enhance_prompt": formdata.get('enhance_prompt', False),
                    "request_type": formdata.get('request_type'),
                    "upscale": formdata.get('upscale', False),
                    "accountId": formdata.get('accountId', None),
                    "inpainting_original_option": formdata.get('inpainting_original_option', False),
                }   
                # if prompt string doesnt end with , and doesnt already have the words "vibrant colours" add it aswell as vibrant colours:
                # if data['prompt'][-1] != ",": 
                #     if "vibrant colours" not in data['prompt']:
                #         data['prompt'] += ", vibrant colours"
                # else:
                #     if "vibrant colours" not in data['prompt']:
                #         data['prompt'] += " vibrant colours"
                    
                
            
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
                        continue  # Skip to the next request
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
                        continue  # Skip to the next request
                else:
                    data['mask_data'] = None
                    
                if data['request_type'] == 'controlnet_img2img':
                    image = np.array(data['image_data'])
                    low_threshold = 100
                    high_threshold = 200
                    image = cv2.Canny(image, low_threshold, high_threshold)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    canny_image = Image.fromarray(image)
                    data['image_data'] = canny_image

                    
                if data['steps'] > 70:
                    if data['image_count'] > 6:
                        data['image_count'] = 1
                    
                    
                    
                if data['seed'] == -1:
                    data['seedNumber'] = random.randint(0, 2**32 - 1)
                else:
                    data['seedNumber'] = data['seed']
                    
                data['seed'] = generator.manual_seed(data['seedNumber'])
                
                # data on multiple print lines for easier debugging
                print("Request ID: " + request_id + "\nModel: " + data['model'] + " | Steps: " + str(data['steps']) + " | Width: " + str(data['width']) + "px | Height: " + str(data['height']) + "px\nSeed: " + str(data['seedNumber']) + " | Strength: " + str(data['strength']) + " | CFGuidance: " + str(data['guidance']) + " | Image Count: " + str(data['image_count'] )  + "\nPrompt: " + data['prompt'] + "\nNegative Prompt: " + data['negative_prompt'] + "\nLora: " + str(data['lora']))

                model_name = data['model']
                lora = data.get('lora', "NO")
                model_type = data['request_type']
                # if model_type isnt inpainting, get_txt2img_model, else get_inpainting_model
                
                # if model_type is txt2img or img2img, get the model, else get the inpainting model:
                if model_type == 'txt2img' or model_type == 'img2img':
                    model = get_txt2img_model(model_name, data)
                elif model_type == 'inpainting':
                    model = get_inpainting_model(model_name, data)
                elif model_type == 'controlnet_img2img':
                    model = get_controlnet_img2img_model(model_name, data)
                elif model_type == 'openpose':
                    model = get_openpose_model(model_name, data)
                    
                # checks the model type to load the correct model:
                if model_type == 'txt2img':
                    current_model = model
                elif model_type == 'img2img':
                    current_model = img2img_models.get(model_name)
                elif model_type == 'inpainting':
                    current_model = model
                elif model_type == 'controlnet_img2img':
                    current_model = model
                elif model_type == 'openpose':
                    current_model = model
                    
                    
                if current_model is not None:
                    if data['lora'] is not None:
                        load_loras(request_id, current_model, lora)
                    model_outputs = process_image(current_model, model_type, data, request_id)
                    if model_outputs == "CONTINUE":
                        error_message = "INPAINTING FAILED"
                        print("Error processing request:", error_message)
                        results[request_id] = {"status": "error", "message": error_message}
                        request_queue.pop(0)
                        continue
                    print("\n")
                    if data['lora'] is not None:
                        current_model.unload_lora_weights()
                        
                    # clear torch memory:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()                    
                    

                    if model_outputs is not None:
                        image_data_list = []
                        for index, img in enumerate(model_outputs):
                            image_data = save_image(request_id, img, model_type, data, image_index=index)
                            image_data_list.append(image_data)

                    results[request_id] = {
                        "images": image_data_list,
                        "additionalInfo": {
                            "seed": data['seedNumber'],
                            "executiontime": time.time() - start_time
                        }
                    }
                else:
                    error_message = "Invalid model name"
                    print("Error processing request:", error_message)
                    results[request_id] = {"status": "error", "message": error_message}

            except Exception as e:
                error_message = str(e)
                print("Error processing request:", error_message)
                results[request_id] = {"status": "error", "message": error_message}

            queue_item.status = "completed"
            clean_completed_requests()
        else:
            time.sleep(1)

def clean_completed_requests():
    """Remove completed requests from the front of the queue."""
    while request_queue and request_queue[0].status == "completed":
        request_queue.pop(0)

# Start a thread that processes requests from the queue
threading.Thread(target=process_requests, daemon=True).start()

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json

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
            
        
            
        # append mask_data and image_data to data
        data['image_data'] = image_data
        data['mask_data'] = mask_data
        
        # validation checks:
        
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
            
        
        request_id = str(uuid.uuid4())
        queue_item = QueueRequest(request_id, data)
        
        # if the request is the exact same as another request (apart from the request_id), remove it, else add it to the queue:
        for index, item in enumerate(request_queue):
            if item.data == queue_item.data:
                error_message = "Duplicate request"
                print(error_message)
                return generate_error_response(error_message, 400)
        else:
            request_queue.append(queue_item)
        
        

        position = len(request_queue)  # Current position in the queue is its length

        return jsonify({"status": "queued", "request_id": request_id, "position": position}), 202

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        return generate_error_response(error_message, 500)  # Return the error response within the request handler

@app.route('/queue_position/<request_id>', methods=['GET'])
def check_queue_position(request_id):
    # Loop through the queue and find the position for the given request_id
    for index, item in enumerate(request_queue):
        if item.request_id == request_id:
            
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1}), 200
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