import base64
from flask import Flask, request, jsonify, send_file
import logging
from flask_cors import CORS
import io
import random
import json
import threading
import requests
import yaml
import uuid
import time
from io import BytesIO
from moviepy.editor import ImageSequenceClip
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import numpy as np
import os
import datetime
import asyncio
import DB
from pathlib import Path

global hash_queue_busy
global hash_queue

hash_queue = []

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
    
    # Model settings
    diffusion_models = {
        'realisticVision': {'model_path': config["realisticVision_model_path"]},
        'fluffysonic': {'model_path': config["fluffysonic_model_path"]},
        'furryblend': {'model_path': config["furryblend_model_path"]},
        'toonify': {'model_path': config["toonify_model_path"]},
        'flat2danimerge': {'model_path': config["flat2danimerge_model_path"]},
        'pdxl-autismmix': {'model_path': config["pdxl_autismmix_model_path"]},
        'pdxl-sonichasautismmix': {'model_path': config["pdxl_sonichasautismmix_model_path"]},
        'pdxl-ponyrealism': {'model_path': config["pdxl_ponyrealism_model_path"]},
        'pdxl-fluffysonic': {'model_path': config["pdxl_fluffysonic_model_path"]},
        'illustrious-wai': {'model_path': config["illustrious_wai_model_path"]},
        'illustrious-novafurry': {'model_path': config["illustrious_novafurry_model_path"]},
        'sd3-medium': {'model_path': config["sd3_medium_model_path"]},
        'flux-unchained': {'model_path': config["flux_unchained_model_path"]},
    }

except Exception as e:
    raise

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

def round_to_multiple_of_eight(number):
    """Round a number to the nearest multiple of 8."""
    # print("Rounding to multiple of 8: ", number)
    return round(number / 8) * 8

app = Flask(__name__)
CORS(app)

@app.route('/get-lora-yaml', methods=['GET'])
def get_lora_yaml():
    return jsonify(lora_weights_map)

def add_watermark(image, text, data):
    
    extras = data['extras']
    
    if(extras["removeWatermark"] == True):
        return image
    
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
    image = image.convert('RGBA')
    
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

def load_loras(request_id, lora_items, data):
    
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
        
    data['loras_data'] = {}

    for item in lora_items:
        time.sleep(0.01)
        try:
            category, key = item.split('-', 1)
            lora_data = lora_weights_map.get(category, {}).get(item)

            if lora_data:
                print(f"data['lora_strengths'] = {data['lora_strengths']}")
                # append [key, strength] to the lora_strengths list:
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
                    # replace any / with \:
                    data['loras_data'][item] = lora_data['lora'], strength
                    
            else:
                print(f"No data found for {item}")
                data['loras_broken'] = True
        except Exception as e:
            print(f"Error processing item '{item}': {e}")
            log_error(f"Error processing item '{item}': {e}", data)
            
def add_metadata(image, metadata, data):
    meta = PngImagePlugin.PngInfo()
   
    for key, value in metadata.items():
        meta.add_text(str(key), str(value))

    return meta


def save_image(request_id, output_image, model_type, data):
    try:

        # Generate metadata
        metadata = {
            "request_id": request_id,
            "model_type": model_type,
            "prompt": data['prompt'],
            "negative_prompt": str(data['negative_prompt']),
            "loras": data['lora'],
            "lora_strengths": data['lora_strengths'],
            "steps": data['steps'],
            "seed": data['seedNumber'],
            "CFG": data['guidance'],
            "model": data['og_model'],
            "generation_date": datetime.datetime.now().isoformat(),
            "accountId": str(data['accountId']),
            "scheduler": data['scheduler']
        }

        # watermarks = ["JSCammie.com", "Cammie.ai", "Check out mobians.ai!", "In femboys we trust", "NeverSFW.gg"]
        # watermarks_chances = [0.5, 0.1, 0.1, 0.1, 0.1]
        # watermark_text = random.choices(watermarks, watermarks_chances)[0]
        watermark_text = "JSCammie.com"
        watermarked_image = add_watermark(output_image, watermark_text, data)
        meta = add_metadata(watermarked_image, metadata, data)
        buffered = io.BytesIO()
        watermarked_image.save(buffered, format="PNG", pnginfo=meta)
        img_str = base64.b64encode(buffered.getvalue()).decode()
            
        if data['save_image'] == True:
            # create dir if you need to:
            os.makedirs("./output_images", exist_ok=True)
            watermarked_image.save(f"./output_images/{request_id}.png", pnginfo=meta)

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

# create websocket client:
import websocket

def check_for_image(prompt_id, data):
    previous_comfyui_response = None
    
    def print_response_if_changed(message, comfyui_response):
        nonlocal previous_comfyui_response
        if comfyui_response != previous_comfyui_response:
            print(message, comfyui_response)
            previous_comfyui_response = comfyui_response
    
    while True:
        time.sleep(0.1)  # Delay to prevent too frequent requests
        try:
            # Fetch history or current status of the image generation using the prompt_id
            response = requests.get(f"{data['server_address']}/history/{prompt_id}")
            comfyui_response = response.json()

            if not comfyui_response:
                continue

            main_data = comfyui_response.get(prompt_id)
            if not main_data:
                continue
            
            # Check for status and error messages
            if main_data['status']['status_str'] == 'error':
                # log the error:
                log_error(f"Error detected in response: {comfyui_response}", data)
                print_response_if_changed("Error detected in response: ", comfyui_response)
                return None
            
            print_response_if_changed("ComfyUI Response: ", comfyui_response)

            # Check if the process is completed
            status_info = main_data.get('status', {})
            if status_info.get('completed', False):
                output_images = []
                image_data = main_data.get('outputs', {})

                # Assume '64' is the key where image data is stored
                image_info_64 = image_data.get('64', {}).get('images', [])
                print("Image Info 64: ", image_info_64)

                for image_info in image_info_64:
                    image_filename = image_info.get('filename')
                    if image_filename:
                        image_path = os.path.join(data['images_dir'], image_filename)
                        try:
                            with Image.open(image_path) as img:
                                output_images.append(img.copy())
                            print(f"Loaded image {image_path}")
                        except IOError as img_error:
                            print(f"Failed to open image {image_path}: {img_error}")
                
                if output_images:
                    print("Images fetched successfully.")
                    return output_images
                else:
                    print("No images were loaded.")
                    return None

        except Exception as e:
            print(f"Error checking for image: {e}")
            return None

    
    
    



def process_request(queue_item):
    try:
        data = queue_item.data
        request_id = queue_item.request_id
        request_json = queue_item.data['request_json']
        
        print("Processing request:", request_id)
        
        start_time = time.time()
        
        # if any loras dont
        if data['loras_broken'] == True:
            print("Broken Lora detected, skipping request.")
            results[request_id] = {"status": "error", "message": "You have a broken lora selected, please refresh the page and try again.\n Failing that, please contact JSCammie!"}
            queue_item.status = "error"
            return "skipped"
        
        p = {"prompt": request_json}
        request_json = json.dumps(p).encode('utf-8')
        
        
        
        comfyui_response_code = None
        
        while comfyui_response_code != 200:
            comfyui_response = requests.post(f"{data['server_address']}/prompt", data=request_json)
            print("ComfyUI Response Status Code: ", comfyui_response.status_code)
            comfyui_response_code = comfyui_response.status_code
            print("ComfyUI Response: ", comfyui_response.text)
            time.sleep(0.5)
            
        prompt_id = comfyui_response.json()['prompt_id']
        
        output_images = check_for_image(prompt_id, data)
        
        # delete the images from the images_dir that match the request_id:
        for filename in os.listdir(data['images_dir']):
            if filename.startswith(f"{request_id}"):
                file_path = os.path.join(data['images_dir'], filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")
        
        if output_images is None:
            results[request_id] = {"status": "error", "message": "Error processing request, outputted images in None, Contact JSCammie!"}
            queue_item.status = "error"
            return "skipped"
        
        

        # Watermark all images:
        # Assume save_image is defined correctly to handle the image processing and saving
        
        Base64Images = []
        for i in range(len(output_images)):
            img = output_images[i]
            watermarked_image = save_image(request_id, img, data['request_type'], data)
            Base64Images.append(watermarked_image)
            
        PIL_Images = []
        
        # convert the base64 images to PIL images:
        for i in range(len(Base64Images)):
            img_str = Base64Images[i]['base64']
            img_data = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_data))
            PIL_Images.append(img)
            
        data['loraStrings'] = ""
        
        # for each lora in the data['lora'], add it to the data['loraStrings']:
        # its an array:
        # if there are loras, the data['lora'] may be empty array OR string, so check if its not empty:
        if data['lora'] != "":
            for index, lora in enumerate(data['lora']):
                data['loraStrings'] += f"{lora} - Strength: {data['loras_data'][lora][1]}\n"
                
        # print("Lora Strings: ", data['loraStrings'])

        hash_object = {
            "data": data,
            "images": PIL_Images,
            "status": "pending"
        }
        hash_queue.append(hash_object)       
            
        time_added_to_queue = data['queued_time']
        total_time = time.time() - time_added_to_queue
        
        average_queue_times.append(total_time)
        
        historyData = {
            "account_id": data['accountId'],
            "prompt": data['prompt'],
            "negative_prompt": data['true_negative_prompt'],
            "model": data['og_model'],
            "aspect_ratio": data['aspect_ratio'],
            "loras": data['lora'],
            "lora_strengths": data['lora_strengths'],
            "steps": data['steps'],
            "cfg": data['guidance'],
            "seed": data['seed'],
            
        }

        results[request_id] = {
            "images": Base64Images,
            "additionalInfo": {
                "seed": data['seed'],
                "executiontime": time.time() - start_time,
                "totaltime": total_time,
                "timestamp": time.time()
            },
            "historyData": historyData,
            "fastqueue": data['fastqueue'],
            "creditsRequired": str(data['creditsRequired']),
        }
        
        queue_item.status = "completed"
        return "processed"

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        queue_item.status = "error"
        results[request_id] = {"status": "error", "message": error_message}

hash_queue_busy = False

async def process_hash_queue():
    global hash_queue_busy
    while True:
        await asyncio.sleep(0.1)  # Corrected to use await
        if not hash_queue_busy:
            if hash_queue:  # Check if the queue is not empty
                hash_item = hash_queue[0]  # Get the first item
                if hash_item['status'] == "pending":
                    hash_queue_busy = True
                    # Assuming DB.process_images_and_store_hashes is an async function
                    # convert queue_item['images'] to a list of PIL images from base 64:
                    await DB.process_images_and_store_hashes(hash_item['images'], hash_item['data'])
                    # print("Finished processing")
                    hash_queue_busy = False
                    hash_queue.remove(hash_item)

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
        # First, check in request_queue_0
        for item in request_queue_0:
            if item.request_id == request_id:
                index = request_queue_0.index(item)
                if index < 2:
                    return jsonify({"status": "processing", "message": "Request is currently being processed"}), 200
                item.status = "cancelled"
                request_queue_0.remove(item)
                return jsonify({"status": "cancelled", "message": "Cancelled Request"}), 200

        # If not found in request_queue_0, check in request_queue_1
        for item in request_queue_1:
            if item.request_id == request_id:
                index = request_queue_1.index(item)
                if index < 2:
                    return jsonify({"status": "processing", "message": "Request is currently being processed"}), 200
                item.status = "cancelled"
                request_queue_1.remove(item)
                return jsonify({"status": "cancelled", "message": "Cancelled Request"}), 200
                    
        return jsonify({"status": "not found", "message": "Invalid request_id, Not Found"}), 404
    
    except Exception as e:
        generate_error_response(str(e), 500)
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route('/queue_position/<request_id>', methods=['GET'])
def check_queue_position(request_id):
    for index, item in enumerate(request_queue_0):
        if item.request_id == request_id:
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": f"{len(request_queue_0)} ({len(request_queue_0) + len(request_queue_1)})"}), 200
    for index, item in enumerate(request_queue_1):
        if item.request_id == request_id:
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": f"{len(request_queue_1)} ({len(request_queue_1) + len(request_queue_0)})"}), 200
    if request_id in results:
        if results[request_id].get("status") == "error":
            return jsonify({"status": "error", "message": results[request_id].get("message")}), 200
        return jsonify({"status": "completed", "request_id": request_id}), 200
    
    return jsonify({"status": "not found", "message": "Invalid request_id"}), 404

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    result = results.get(request_id)
    # print("Results: ", result)
    
    # if the images are PIL then convert them to base64:
    if result:
        # if the type is PngImageFile then convert it to base64:
        if type(result['images'][0]) == PngImagePlugin.PngImageFile:
            for i in range(len(result['images'])):
                img = result['images'][i]
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result['images'][i] = {"base64": img_str, "width": img.width, "height": img.height}
    
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

        # print("Headers:", request.headers)
        # print("Data:", request.get_json())
                
        data['og_model'] = data['model']
        
        if data['model'] == "pdxl-ponydiffusion":
            data['model'] = "pdxl-autismmix"
            
            
        if data['model'] == "flux-unchained":
            flux_count = 0
            if str(data['accountId']) != "1039574722163249233":
                # if there are any flux requests in the queue, return an error:
                for item in request_queue_0:
                    if item.data['model'].startswith("flux-"):
                        flux_count += 1
                for item in request_queue_1:
                    if item.data['model'].startswith("flux-"):
                        flux_count += 1
                        
                if flux_count > 3:
                    return generate_error_response("Flux unchained is currently disabled, please consider Buying Credits to help support the developer of the website @ enable me to upgrade the server so Flux can be unlimited again.", 503)
        
        if global_settings.get('pdxl', False):
            if data['model'].startswith("pdxl-"):
                return generate_error_response("PDXl Models are currently disabled, please use the other models instead.", 503)
            
        if global_settings.get('disable_illustrious', False):
            if data['model'].startswith("illustrious-"):
                return generate_error_response("Illustrious Models are currently disabled, please use the other models instead.", 503)
            
        if data['request_type'] == "txt2video":
            return generate_error_response("Text to video is currently disabled.", 503)
        
        data['gpu_id'] = 0
        
        if len(request_queue_0) > len(request_queue_1) + 1:
            data['gpu_id'] = 1
            
        if len(request_queue_1) > len(request_queue_0) + 1:
            data['gpu_id'] = 0
                    
        if data['model'].startswith("pdxl-"):
            if data['steps'] > global_settings['pdxl_max_steps']:
                data['steps'] = global_settings['pdxl_max_steps']
                
        if data['model'].startswith("illustrious-"):
            if data['steps'] > global_settings['illustrious_max_steps']:
                data['steps'] = global_settings['illustrious_max_steps']
                
        if data.get('aspect_ratio', None) is not None:
            if data['aspect_ratio'] == "portrait":
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 768
                    data['height'] = 1024
                else:
                    data['width'] = 512
                    data['height'] = 768
            elif data['aspect_ratio'] == "landscape":
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 1024
                    data['height'] = 768
                else:
                    data['width'] = 768
                    data['height'] = 512
                    
                
            elif data['aspect_ratio'] == "square":
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 1024
                    data['height'] = 1024
                else:
                    data['width'] = 512
                    data['height'] = 512
            # else:
            #     return generate_error_response("Invalid aspect ratio, please choose from 'portrait', 'landscape', or 'square'.", 400)
                
            elif data['aspect_ratio'] == "bannerHorizontal":
                data['width'] = 1024
                data['height'] = 512
                
            elif data['aspect_ratio'] == "bannerVertical":
                data['width'] = 512
                data['height'] = 1024
                
            elif data['aspect_ratio'] == "16:9":
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 1024
                    data['height'] = 576
                else:
                    data['width'] = 768
                    data['height'] = 432
                
            elif data['aspect_ratio'] == "9:16":
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 576
                    data['height'] = 1024
                else:
                    data['width'] = 432
                    data['height'] = 768
            
            elif data['aspect_ratio'] == "21:9":
                # needs to be divisible by 64:
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 1344
                    data['height'] = 576
                else:
                    data['width'] = 1024
                    data['height'] = 432
            elif data['aspect_ratio'] == "9:21":
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    data['width'] = 576
                    data['height'] = 1344
                else:
                    data['width'] = 432
                    data['height'] = 1024
                
            if data['model'].startswith("pdxl-") or data['model'].startswith("flux-"):
                # multiply the width and height by the global_settings['pdxl_resolution_multiplier']:
                data['width'] = data['width'] * global_settings['pdxl_resolution_multiplier']
                data['height'] = data['height'] * global_settings['pdxl_resolution_multiplier']
                
            if data['model'].startswith("illustrious-"):
                data['width'] = data['width'] * global_settings['illustrious_resolution_multiplier']
                data['height'] = data['height'] * global_settings['illustrious_resolution_multiplier']
                
            if data['model'].startswith("sd3-"):
                # multiply the width and height by the global_settings['sd3_resolution_multiplier']:
                data['width'] = data['width'] * global_settings['sd3_resolution_multiplier']
                data['height'] = data['height'] * global_settings['sd3_resolution_multiplier']
                
                
                
            else:
                data['width'] = data['width'] * global_settings['sd15_resolution_multiplier']
                data['height'] = data['height'] * global_settings['sd15_resolution_multiplier']
                
        if data['model'].startswith("pdxl-"):
            if len(data['lora']) > global_settings['max_loras_pdxl']:
                return generate_error_response(f"Maximum number of Lora options for PDXL models is: {global_settings['max_loras_pdxl']}", 400)
        elif data['model'].startswith("flux-"):
            if len(data['lora']) > global_settings['max_loras_flux']:
                return generate_error_response(f"Maximum number of Lora options for Flux models is: {global_settings['max_loras_flux']}", 400)
        elif data['model'].startswith("illustrious-"):
            if len(data['lora']) > global_settings['max_loras_illustrious']:
                return generate_error_response(f"Maximum number of Lora options for Illustrious models is: {global_settings['max_loras_illustrious']}", 400)
        else:
            if len(data['lora']) > global_settings['max_loras']:
                return generate_error_response(f"Maximum number of Lora options is: {global_settings['max_loras']}", 400)
        
        data['width'] = round_to_multiple_of_eight(data['width'])
        data['height'] = round_to_multiple_of_eight(data['height'])
        
        
        
        # upscale global_settings sets:
        
        data['upscale_steps'] = global_settings['upscale_steps']
        data['upscale_denoise'] = global_settings['upscale_denoise']
        data['upscale_scale'] = global_settings['upscale_scale']
        
        
        
        

        # Validate and preprocess the input data
        validate_start_time = time.time()
        true_prompt = data['prompt']
        
        # remove excess commas:
        data['prompt'] = data['prompt'].replace(",,", ",")
        data['prompt'] = data['prompt'].replace(", ,", ",")
        data['prompt'] = data['prompt'].replace(",,", ",")
        data['prompt'] = data['prompt'].replace(", ,", ",")
        data['prompt'] = data['prompt'].replace(",,", ",")
        data['prompt'] = data['prompt'].replace(", ,", ",")

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

        data['accountId'] = data.get('accountId', "0")

        if data['accountId'] == "":
            data['accountId'] = "0"

        data['accountId'] = str(data['accountId'])

        data['prompt'] = randomize_string(data['prompt'])

        if str(data['prompt']) == "{'status': 'error', 'message': 'Mismatched brackets'}":
            return None, "Mismatched brackets ('{}' brackets are used to denote a random choice, and must be used in pairs, here is an example of a correct usage: '{woman|man} with {long|short} hair')"

        if data['steps'] > 126:
            return None, "You have reached the limit of 125 steps per request. Please reduce the number of steps and try again."

        if data['quantity'] > 5:
            return None, "You have reached the limit of 4 images per request. Please reduce the number of images and try again."

        if len(data['lora']) > 5:
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
            
        if data['request_type'] != 'txt2img' and data['request_type'] != 'inpainting' and data['request_type'] != 'img2img':
            return None, "Invalid request type. Only txt2img, img2img and inpainting requests are allowed at this time."

        model = data.get("model", "sonic")
        
        if model == None:
            return None, "Model is not specified, please try re-selecting the model and try again."
        
        data['true_negative_prompt'] = data.get("negativeprompt", "")

        if data.get("model", "sonic").startswith("pdxl-") or data.get("model", "sonic").startswith("flux-") or data.get("model", "sonic").startswith("illustrious-"):
            
            negative_embedding_words_pdxl = ""
            negative_prompt_final = negative_embedding_words_pdxl + data.get("negativeprompt", "")
            
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
        
        data['input_image'] = data.get("input_image", None)
        
        if data['input_image'] is not None:
            # load the image from the file specified in the input_image field:
            try:
                data['image'] = Image.open(data['input_image'])
            except Exception as e:
                return generate_error_response("Failed to identify image file", 400)

        if data['image'] is not None:
            try:
                if data['input_image'] is None:
                    base64_encoded_data = data['image'].split(',', 1)[1]
                    image_data = base64.b64decode(base64_encoded_data)
                    img_bytes = io.BytesIO(image_data)
                    data['image'] = Image.open(img_bytes)
                
                # print("Image width height before")
                
                # Determine the scaling factor to ensure both sides are at least 512px OR 768px for pdxl/flux models
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    scale_factor = max(768 / data['image'].width, 768 / data['image'].height)
                else:
                    scale_factor = max(512 / data['image'].width, 512 / data['image'].height)
                
                # print("Image width height after")

                # Calculate new dimensions
                new_width = round_to_multiple_of_eight(data['image'].width * scale_factor)
                new_height = round_to_multiple_of_eight(data['image'].height * scale_factor)
                
                # if they are perfect squares of 512 or 768 then set to be 768 or 1024:
                if new_width == new_height:
                    if new_width == 512:
                        new_width = 768
                        new_height = 768
                    elif new_width == 768:
                        new_width = 1024
                        new_height = 1024
                
                # Update dimensions in the data dictionary
                data['width'], data['height'] = new_width, new_height
                    

                # Resize the image
                data['image'] = data['image'].resize((new_width, new_height))
                data['image'] = data['image'].convert('RGBA')


            except Exception as e:
                return generate_error_response("Failed to identify image file", 400)
        else:
            data['image'] = None
            
        if data['inpaintingMask'] is not None:
            try:
                base64_encoded_data = data['inpaintingMask'].split(',', 1)[1]
                image_data = base64.b64decode(base64_encoded_data)
                img_bytes = io.BytesIO(image_data)
                data['inpaintingMask'] = Image.open(img_bytes)
            
                # print("Image width height before")
                
                # Determine the scaling factor to ensure both sides are at least 512px OR 768px for pdxl/flux models
                if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-"):
                    scale_factor = max(768 / data['inpaintingMask'].width, 768 / data['inpaintingMask'].height)
                else:
                    scale_factor = max(512 / data['inpaintingMask'].width, 512 / data['inpaintingMask'].height)
                
                
                # print("Image width height after")

                # Calculate new dimensions
                new_width = round_to_multiple_of_eight(data['inpaintingMask'].width * scale_factor)
                new_height = round_to_multiple_of_eight(data['inpaintingMask'].height * scale_factor)
                
                # if they are perfect squares of 512 or 768 then set to be 768 or 1024:
                if new_width == new_height:
                    if new_width == 512:
                        new_width = 768
                        new_height = 768
                    elif new_width == 768:
                        new_width = 1024
                        new_height = 1024


                # Update dimensions in the data dictionary
                data['width'], data['height'] = new_width, new_height

                # Resize the inpaintingMask
                data['inpaintingMask'] = data['inpaintingMask'].resize((new_width, new_height))
                data['inpaintingMask'] = data['inpaintingMask'].convert('RGB')


            except Exception as e:
                return generate_error_response("Failed to identify inpaintingMask file", 400)
        else:
            data['inpaintingMask'] = None
            
            
            
        data['regionalPromptSettings'] = data.get("regionalPromptSettings", {"status": "false"})
    
        if data['regionalPromptSettings']['status'] == "true":
            
            # print("Regional Prompt Settings: ", data['regionalPromptSettings'])
            data['regionalPromptSettings']['regionalPromptSplitPosition'] = float(data['regionalPromptSettings']['regionalPromptSplitPosition'])
            data['regionalPromptSettings']['regionalPromptAStrength'] = float(data['regionalPromptSettings']['regionalPromptAStrength'])
            data['regionalPromptSettings']['regionalPromptBStrength'] = float(data['regionalPromptSettings']['regionalPromptBStrength'])
            
            image = Image.new('RGB', (data['width'], data['height']), color = (0, 0, 0))
            
            # split the image into two parts:
            split_position = int(data['regionalPromptSettings']['regionalPromptSplitPosition'] * data['width'] / 100)
            
            # create the two images have one side be blue and the other red, set the regionalPromptSettingsHex1 and regionalPromptSettingsHex2 to be the hex values:
            data['regionalPromptSettings']['hexA'] = "#0000FF"
            data['regionalPromptSettings']['hexB'] = "#FF0000"
            
            leftImage = Image.new('RGB', (int(split_position), data['height']), color = data['regionalPromptSettings']['hexA'])
            rightImage = Image.new('RGB', (data['width'] - int(split_position), data['height']), color = data['regionalPromptSettings']['hexB'])
            
            image.paste(leftImage, (0, 0))
            image.paste(rightImage, (int(split_position), 0))
            
            data['image'] = image
            
            
            
        og_seed = data['seed']

        if int(data['seed']) == -1:
            data['seedNumber'] = random.randint(0, 2**32 - 1)
        else:
            data['seedNumber'] = int(data['seed'])
            
        data['seed'] = data['seedNumber']
        
        # get all the red pixels from the inpaintingMask image and place them ontop of a transparent image:
        if data['inpaintingMask'] is not None:
            # use the inpaintingMask for width and height:
            transparent_image = Image.new('RGBA', (data['inpaintingMask'].width, data['inpaintingMask'].height), (0, 0, 0, 0))
            # get the red pixels from the inpaintingMask image:
            red_pixels = data['inpaintingMask'].convert("RGBA")
            red_data = red_pixels.getdata()
            new_data = []
            for item in red_data:
                # get all red pixels:
                if item[0] == 255 and item[1] == 0 and item[2] == 0:
                    new_data.append((255, 0, 0, 255))
                else:
                    new_data.append((0, 0, 0, 0))
            transparent_image.putdata(new_data)
            data['inpaintingMask'] = transparent_image
            

        validated_data = {
            'model': data.get('model'),
            'prompt': data.get('prompt'),
            'negative_prompt': negative_prompt_final,
            'true_negative_prompt':  data['true_negative_prompt'],
            'image_count': int(data.get("quantity")),
            'steps': int(data.get("steps", 20)),
            'width': int(data.get("width", 512)),
            'height': int(data.get("height", 512)),
            'aspect_ratio': str(data.get("aspect_ratio")),
            'seed': int(data.get("seed", -1)),
            'strength': float(data.get("strength", 0.75)),
            'guidance': float(data.get("guidance", 5)),
            'image_data': data.get("image", None),
            'mask_data': data.get("inpaintingMask", None),
            'lora': data.get('lora', None),
            'lora_strengths': data.get('lora_strengths', None),
            'enhance_prompt': data.get('enhance_prompt', False),
            'request_type': data['request_type'],
            'inpainting_original_option': True,
            'splitType': data.get('splitType', "horizontal"),
            'splits': int(data.get('splits', 1)),
            'splitOverlap': float(data.get('splitOverlap', 0.1)),
            'finalStrength': float(data.get('finalStrength', 0.2)),
            'video_length': int(data.get('video_length', 16)),
            'accountId': str(data.get('accountId', "0")),
            'true_prompt': str(true_prompt),
            'scheduler': data.get('scheduler', "eulera"),
            'seedNumber': int(data['seedNumber']),
            'og_seed': int(og_seed),
            "save_image": bool(data.get("save_image", False)),
            "gpu_id": int(data.get("gpu_id", 0)),
            "ip": data.get("ip", ""),
            "userAgent": data.get("userAgent", ""),
            "queued_time": validate_start_time,
            "image": data['image'],
            "lightning_mode": data.get("lightning_mode", False),
            "og_model": data['og_model'],
            "fastqueue": data.get("fastqueue", False),
            "creditsRequired": data.get("creditsRequired", 0),
            "extras": data.get("extras", {"removeWatermark": False, "upscale": False, "doubleImages": False, "removeBackground": False}),
            "regionalPromptSettings": data.get("regionalPromptSettings", {"status": "false"}),
            "upscale_steps": data.get("upscale_steps", global_settings['upscale_steps']),
            "upscale_denoise": data.get("upscale_denoise", global_settings['upscale_denoise']),
            "upscale_scale": data.get("upscale_scale", global_settings['upscale_scale']),
        }
        
        # sanity check that each extras values are there, if they arent then add them as False:
        if "removeWatermark" not in validated_data['extras']:
            validated_data['extras']['removeWatermark'] = False
            
        if "upscale" not in validated_data['extras']:
            validated_data['extras']['upscale'] = False
            
        if "doubleImages" not in validated_data['extras']:
            validated_data['extras']['doubleImages'] = False
            
        if "removeBackground" not in validated_data['extras']:
            validated_data['extras']['removeBackground'] = False
            
        if validated_data['extras']['doubleImages'] == True:
            validated_data['image_count'] = validated_data['image_count'] * 2
            
            
            
        if validated_data['regionalPromptSettings']['status'] == "true":
            # split the prompt into 3 parts, the first part is the prompt, the second part is the regional promptA, the third part is the regional promptB:
            prompt_split = validated_data['prompt'].split("<rp>")
            validated_data['regionalPromptSettings']['regionalPromptBase'] = prompt_split[0]
            validated_data['regionalPromptSettings']['regionalPromptA'] = prompt_split[1]
            validated_data['regionalPromptSettings']['regionalPromptB'] = prompt_split[2]
            
            if validated_data['extras']['upscale'] == False:
                # set steps to be 1.5x the steps:
                validated_data['steps'] = int(validated_data['steps'] * 1.5)
                # set width and height to be 1.3x the width and height:
                validated_data['width'] = int(validated_data['width'] * 1.3)
                validated_data['height'] = int(validated_data['height'] * 1.3)
            else: 
                # set steps to be 1.3x the steps:
                validated_data['steps'] = int(validated_data['steps'] * 1.3)
            
        
        data['width'] = round_to_multiple_of_eight(data['width'])
        data['height'] = round_to_multiple_of_eight(data['height'])
        

        # Check for banned users and fast pass
        account_id = validated_data['accountId']
        request_type = validated_data['request_type']
        
        if request_type == "latent_couple":
            return generate_error_response("Latent couple is currently disabled.", 503)
        
        if request_type == "openpose":
            return generate_error_response("Openpose is currently disabled.", 503)

        if account_id is not None:
            ban_check_result = check_banned_users(account_id, request_type)
            if ban_check_result is not False:
                return generate_error_response(f"User {account_id} is banned for {ban_check_result['reason']}, in the {request_type} category", 400)
            
        if str(account_id) != "0" and str(account_id):
            # if account_id is already in the request_queue_0 or request_queue_1, return an error:
            if data['gpu_id'] == 0:
                for item in request_queue_0:
                    if str(item.data['accountId']) == str(account_id):
                        # send them the correct info:
                        # return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202
                        # send them info about their request to allow the client to resume it:
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_0.index(item) + 1, "queue_length": len(request_queue_0)}), 202

            elif data['gpu_id'] == 1:
                for item in request_queue_1:
                    if str(item.data['accountId']) == str(account_id):
                        # send them the correct info:
                        # return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202
                        # send them info about their request to allow the client to resume it:
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_1.index(item) + 1, "queue_length": len(request_queue_1)}), 202
                        
        if data['ip'] is not None:
            if data['ip'] != "123123123":
                for item in request_queue_0:
                    if item.data['ip'] == data['ip']:
                        # send them the correct info:
                        # return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202
                        # send them info about their request to allow the client to resume it:
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_0.index(item) + 1, "queue_length": len(request_queue_0)}), 202
                        
                for item in request_queue_1:
                    if item.data['ip'] == data['ip']:
                        # send them the correct info:
                        # return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202
                        # send them info about their request to allow the client to resume it:
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_1.index(item) + 1, "queue_length": len(request_queue_1)}), 202
        
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
                
        # Check if the model is valid
        model_name = validated_data['model']
        if model_name not in diffusion_models:
            return generate_error_response("Invalid model name", 400)

        request_id = str(uuid.uuid4())
        
        if data['inpaintingMask'] is not None:
            # check that dir "process_images\inpainting" exists, if not create it:
            if not os.path.exists("process_images/inpainting"):
                os.makedirs("process_images/inpainting")
            data['inpaintingMask'].save(f"process_images/inpainting/inpainting{request_id}{data['gpu_id']}.png")
            
        if data['image'] is not None:
            # check that dir "process_images\image" exists, if not create it:
            if not os.path.exists("process_images/image"):
                os.makedirs("process_images/image")
                
            data['image'].save(f"process_images/image/image{request_id}{data['gpu_id']}.png")
            
        data = validated_data
        
        data['request_id'] = request_id
        
        
        
        
        ckpt_name = diffusion_models[data['model']]['model_path']
        
        # remove models/ from the ckpt_name:
        ckpt_name = ckpt_name.replace("models/", "")
        
        # load it from the json:
        if data['request_type'] == "txt2img":
            if data['model'].startswith("sd3"):
                with open ('SD3-txt2imgv2-1.json', 'r') as f:
                    # make it a dict:
                    request_json = json.load(f)
            else:
                with open('txt2imgv3.json', 'r') as f:
                    # make it a dict:
                    request_json = json.load(f)
        elif data['request_type'] == "img2img":
            with open('img2imgv3.json', 'r') as f:
                # make it a dict:
                request_json = json.load(f)
        elif data['request_type'] == "inpainting":
            with open('inpaintingv3.json', 'r') as f:
                # make it a dict:
                request_json = json.load(f)
                
        # print("Request JSON 288: ", request_json['288'])
        request_json['288']['inputs']['ckpt_name'] = ckpt_name
                
        if data['request_type'] == "img2img" or data['request_type'] == "inpainting":
            request_json['301']['inputs']['value'] = data['strength']
            # convert the PIL Image to a base64 string:
            # print type:
            # make sure the directory exists:
            os.makedirs("process_images/image", exist_ok=True)
            request_json['307']['inputs']['image'] = f"C:\\Users\\anime\\Documents\\Coding\\JSCammie-SD-API\\process_images\\image\\image{request_id}{data['gpu_id']}.png"
        else:
            # txt2img, so set the strength to 1:
            request_json['301']['inputs']['value'] = 1
            
        if data['request_type'] == "inpainting":
            # make sure the dir exists:
            os.makedirs("process_images/inpainting", exist_ok=True)
            request_json['309']['inputs']['image'] = f"C:\\Users\\anime\\Documents\\Coding\\JSCammie-SD-API\\process_images\\inpainting\\inpainting{request_id}{data['gpu_id']}.png"
        
        if data['lora_strengths'] is None or data['lora_strengths'] == []:
            data['lora_strengths'] = []
            
        data['loras_broken'] = False
        
        load_loras(request_id, data['lora'], data)
        
        for key in data['loras_data']:
            print(key, data['loras_data'][key])
            
            # for each lora in the data['loras_data'] dict, add it to the request_json:
            for i, key in enumerate(data['loras_data']):
                request_json['117']['inputs'][f"lora_{i+1}"] = {
                    "on": True,
                    "lora": data['loras_data'][key][0],
                    "strength": data['loras_data'][key][1]
                }
                        
        # Prompt
        request_json['298']['inputs']['text'] = data['prompt']
        
        # Negative Prompt
        request_json['300']['inputs']['text'] = data['negative_prompt']
        
        # Steps
        request_json['303']['inputs']['value'] = data['steps']
        request_json['305']['inputs']['value'] = data['guidance']
        request_json['304']['inputs']['value'] = data['seed']
        
        if data['request_type'] == "txt2img":
            request_json['285']['inputs']['width'] = data['width']
            request_json['285']['inputs']['height'] = data['height']
            
        request_json['306']['inputs']['value'] = data['image_count']
            
            
            
        request_json['64']['inputs']['filename_prefix'] = request_id
            
        if not data['model'].startswith("pdxl") and not data['model'].startswith("flux") and not data['model'].startswith("illustrious"):
            request_json['300']['inputs']['text'] = f"{request_json['300']['inputs']['text']} embedding:boring_e621_v4.pt embedding:fluffynegative.pt embedding:badyiffymix41.safetensors embedding:gnarlysick-neg.pt embedding:negative_hand-neg.pt"
                    
        # print(f"LIGHNING MODE: {data['lightning_mode']}")
                    
        if data['model'].startswith("flux"):
                request_json['282']['inputs']['sampler_name'] = "euler"
                request_json['282']['inputs']['scheduler'] = "normal"
        else:
            if data['scheduler'] == "eulera":
                request_json['282']['inputs']['sampler_name'] = "euler_ancestral"
                request_json['282']['inputs']['scheduler'] = global_settings['ksampler_scheduler']
            if data['scheduler'] == "dpm":
                request_json['282']['inputs']['sampler_name'] = "dpmpp_2m"
                request_json['282']['inputs']['scheduler'] = global_settings['ksampler_scheduler']
            if data['scheduler'] == "ddim":
                request_json['282']['inputs']['sampler_name'] = "ddim"
                request_json['282']['inputs']['scheduler'] = global_settings['ksampler_scheduler']
                
                
                
                
                
                
                
        regionalPromptSettings = data['regionalPromptSettings']
        
        if regionalPromptSettings['status'] == "true":
            
            highest_node_id = 0
            for key in request_json:
                # check if its an int before comparing:
                if key.isnumeric():
                    if int(key) > highest_node_id:
                        highest_node_id = int(key)
                        
            # add 1 to the highest node ID to get the new node ID:
            regionalPromptNodes = {
                "image": str(highest_node_id + 1),
                "promptAText": str(highest_node_id + 2),
                "promptBText": str(highest_node_id + 3),
                "promptAPrompt": str(highest_node_id + 4),
                "promptBPrompt": str(highest_node_id + 5),
                "conditioning": str(highest_node_id + 6)
            }
            
            regionalPromptImageNode = {
                "inputs": {
                "image": f"C:\\Users\\anime\\Documents\\Coding\\JSCammie-SD-API\\process_images\\image\\image{request_id}{data['gpu_id']}.png",
                "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                "title": "Load Image"
                }
            } 
            
            regionalPromptNodeAText = {
                "inputs": {
                    "text": f"{regionalPromptSettings['regionalPromptBase']}, {regionalPromptSettings['regionalPromptA']}"
                },
                "class_type": "JWStringMultiline",
                "_meta": {
                    "title": "Positive Prompt A"
                }
            }

            regionalPromptNodeBText = {
                "inputs": {
                    "text": f"{regionalPromptSettings['regionalPromptBase']}, {regionalPromptSettings['regionalPromptB']}"
                },
                "class_type": "JWStringMultiline",
                "_meta": {
                    "title": "Positive Prompt B"
                }
            }

            
            regionalPromptNodeAPrompt = {
                "inputs": {
                "mask_color": regionalPromptSettings['hexA'],
                "strength": float(regionalPromptSettings['regionalPromptAStrength'] / 100),
                "set_cond_area": "default",
                "prompt": [
                    regionalPromptNodes['promptAText'],
                    0
                ],
                "dilation": 0,
                "clip": [
                    "117",
                    1
                ],
                "color_mask": [
                    regionalPromptNodes['image'],
                    0
                ]
                },
                "class_type": "RegionalConditioningColorMask //Inspire",
                "_meta": {
                "title": "Regional Conditioning By Color Mask (Inspire)"
                }
            }
            
            regionalPromptNodeBPrompt = {
                "inputs": {
                "mask_color": regionalPromptSettings['hexB'],
                "strength": float(regionalPromptSettings['regionalPromptBStrength'] / 100),
                "set_cond_area": "default",
                "prompt": [
                    regionalPromptNodes['promptBText'],
                    0
                ],
                "dilation": 0,
                "clip": [
                    "117",
                    1
                ],
                "color_mask": [
                    regionalPromptNodes['image'],
                    0
                ]
                },
                "class_type": "RegionalConditioningColorMask //Inspire",
                "_meta": {
                "title": "Regional Conditioning By Color Mask (Inspire)"
                }
            }
            
            regionalPromptNodeConditioning = {
                "inputs": {
                "conditioning_1": [
                    regionalPromptNodes['promptAPrompt'],
                    0
                ],
                "conditioning_2": [
                    regionalPromptNodes['promptBPrompt'],
                    0
                ]
                },
                "class_type": "ConditioningCombine",
                "_meta": {
                "title": "Conditioning (Combine)"
                }
            }
            
            # set 282 'positive' to the regionalPromptNodeConditioning:
            request_json['282']['inputs']['positive'] = [regionalPromptNodes['conditioning'], 0]
            
            # add all the regional prompt nodes to the request_json:
            request_json[regionalPromptNodes['image']] = regionalPromptImageNode
            request_json[regionalPromptNodes['promptAText']] = regionalPromptNodeAText
            request_json[regionalPromptNodes['promptBText']] = regionalPromptNodeBText
            request_json[regionalPromptNodes['promptAPrompt']] = regionalPromptNodeAPrompt
            request_json[regionalPromptNodes['promptBPrompt']] = regionalPromptNodeBPrompt
            request_json[regionalPromptNodes['conditioning']] = regionalPromptNodeConditioning
            
                
        extras = data['extras']
        
        if extras['upscale'] == True:
            
            # detect highest node ID in the request_json:
            highest_node_id = 0
            for key in request_json:
                # check if its an int before comparing:
                if key.isnumeric():
                    if int(key) > highest_node_id:
                        highest_node_id = int(key)
                        
            # add 1 to the highest node ID to get the new node ID:
            upscale_scheduler = highest_node_id + 1
            upscale_vaedecode = highest_node_id + 2
            upscale_imagescaleby = highest_node_id + 3
            upscale_vaeencode = highest_node_id + 4
            
            # find the ID that feeds into 64
            feedInto64 = request_json['64']['inputs']['images'][0]
            feedInto64 = int(feedInto64)
            
            upscale_vaedecodeNode = {
                "inputs": {
                    "samples": ["282", 0],
                    "vae": ["288", 2]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            }
            
            upscale_imagescalebyNode = {
                "inputs": {
                    "upscale_method": "nearest-exact",
                    "scale_by": data['upscale_scale'],
                    "image": [str(upscale_vaedecode), 0]
                },
                "class_type": "ImageScaleBy",
                "_meta": {
                    "title": "Upscale Image By"
                }
            }
            
            upscale_vaeencodeNode = {
                "inputs": {
                    "pixels": [str(upscale_imagescaleby), 0],
                    "vae": ["288", 2]
                },
                "class_type": "VAEEncode",
                "_meta": {
                    "title": "VAE Encode"
                }
            }
            
            # add the new nodes to the request_json, base it off of 282:
            upscale_schedulerNode = {
                "inputs": {
                    "seed": request_json['282']['inputs']['seed'],
                    "steps": data['upscale_steps'],
                    "cfg": ["305", 0],
                    "sampler_name": request_json['282']['inputs']['sampler_name'],
                    "scheduler": request_json['282']['inputs']['scheduler'],
                    "denoise": data['upscale_denoise'],
                    "model": ["117",0],
                    "positive": ["294", 0],
                    "negative": ["295", 0],
                    "latent_image": [str(upscale_vaeencode), 0]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            }
            
            # add the new nodes to the request_json:
            request_json[str(upscale_scheduler)] = upscale_schedulerNode
            request_json[str(upscale_vaedecode)] = upscale_vaedecodeNode
            request_json[str(upscale_imagescaleby)] = upscale_imagescalebyNode
            request_json[str(upscale_vaeencode)] = upscale_vaeencodeNode
            
            # set 141 vae decode to use the new upscale_scheduler as an inputs images:
            request_json['141']['inputs']['samples'] = [str(upscale_scheduler), 0]
                        
        if extras['removeBackground'] == True:
            
            # detect highest node ID in the request_json:
            highest_node_id = 0
            
            for key in request_json:
                # check if its an int before comparing:
                if key.isnumeric():
                    if int(key) > highest_node_id:
                        highest_node_id = int(key)
                        
            # add 1 to the highest node ID to get the new node ID:
            removeBackgroundImageNodeId = highest_node_id + 1
            
            # find the ID that feeds into 64
            
            feedInto64 = request_json['64']['inputs']['images'][0]
            feedInto64 = int(feedInto64)
            
            # add the new node to the request_json:
            removeBackgroundImageNode = {
                "inputs": {
                    "rem_mode": "Inspyrenet",
                    "image_output": "Hide",
                    "save_prefix": "ComfyUI",
                    "torchscript_jit": False,
                    "images": [
                        str(feedInto64),
                        0
                    ]
                },
                "class_type": "easy imageRemBg",
                "_meta": {
                    "title": "Image Remove Bg"
                }
            }
            
            # add the removeBackgroundImageNode to the request_json, with the new node ID:
            request_json[str(removeBackgroundImageNodeId)] = removeBackgroundImageNode
            
            # set 64 to use the new removeBackgroundImageNodeId as an inputs images:
            request_json['64']['inputs']['images'] = [str(removeBackgroundImageNodeId), 0]
        
        data['request_json'] = request_json
        
        queue_item = QueueRequest(request_id, data)

        # Check for duplicate requests
        if data['gpu_id'] == 0:
            if queue_item.data in [item.data for item in request_queue_0]:
                return generate_error_response("Duplicate request", 400)
        elif data['gpu_id'] == 1:
            if queue_item.data in [item.data for item in request_queue_1]:
                return generate_error_response("Duplicate request", 400)
        
        fastqueue = None
        queueNumber = None
        
        data['server_address'] = "127.0.0.1:8188"

        if data['gpu_id'] == 0:
            queueNumber = 0
            data['server_address'] = "http://127.0.0.1:8188"
            data['images_dir'] = "C:/Users/anime/Documents/Coding/ComfyUIs/ComfyUI/output"
            position = len(request_queue_0)  # Current position in the queue is its length
            
        elif data['gpu_id'] == 1:
            queueNumber = 1
            data['server_address'] = "http://127.0.0.1:8188"
            data['images_dir'] = "C:/Users/anime/Documents/Coding/ComfyUIs/ComfyUI/output"
            # data['server_address'] = "http://127.0.0.1:8189"
            # data['images_dir'] = "C:/Users/anime/Documents/Coding/Stability Matrix/Data/Packages/ComfyUI2/output"
            position = len(request_queue_1)  # Current position in the queue is its length
            
                
        if data['fastqueue'] is True:
            fastqueue = True
        
        positionToSet = 0
        
        if fastqueue == True:
            if queueNumber == 0:
                # if someone is already using the fastqueue, then make their position above the fastqueue'er(s):
                for item in request_queue_0:
                    if item.data['fastqueue'] == True:
                        positionToSet += 1
                    else:
                        break

                request_queue_0.insert(positionToSet, queue_item)
            
            elif queueNumber == 1:
                # if someone is already using the fastqueue, then make their position above the fastqueue'er(s):
                for item in request_queue_1:
                    if item.data['fastqueue'] == True:
                        positionToSet += 1
                    else:
                        break

                request_queue_1.insert(positionToSet, queue_item)
                
        else:
            if queueNumber == 0:
                request_queue_0.append(queue_item)
            elif queueNumber == 1:
                request_queue_1.append(queue_item)
                        
                        
        
        

        return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202

    except Exception as e:
        error_message = str(e)
        print("Error processing request:", error_message)
        # print(data)
        return generate_error_response(error_message, 500)  # Return the error response within the request handler

generateTestJson6 = {
    "model": "fluffysonic",
    "prompt": "((Masterpiece)), high quality, studio quality, intricate details, 4k, solo, (emphasis lines), 2d, cartoon_portrait, 1girl, amy rose, green eyes, pink fur, nude, thick, sexy, at beach, facing viewer, afrobull, by akami mirai",
    "steps": 25,
    "width": 512,
    "height": 512,
    "seed": -1,
    "quantity": 1,
    "request_type": "txt2img",
    "lora": ['style-afrobull', 'style-akamimirai'],
    "lora_strengths": [0.5, 1.2],
    "ip": "123123123"
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
    
test_generate_image(generateTestJson6)



    
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

    asyncio.run(main())
    print("Startup time: " + str(time.time() - program_start_time) + " seconds")
    app.run(host='127.0.0.1', port=5003)