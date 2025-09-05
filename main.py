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
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import numpy as np
import os
import shutil
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

def clear_processed_files(processed_files, output_path):
    """Clear specific files that were processed by this application from output folder."""
    cleared_count = 0
    for filename in processed_files:
        try:
            file_path = os.path.join(output_path, filename)
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"Cleared processed file: {filename}")
                cleared_count += 1
        except Exception as e:
            print(f"Failed to delete processed file {filename}: {e}")
    
    if cleared_count > 0:
        print(f"Successfully cleared {cleared_count} processed files from output folder")

def clear_temp_folder():
    """Completely clear ComfyUI temp folder."""
    temp_path = "C:/Users/anime/Documents/Coding/ComfyUIs/ComfyUI-P/ComfyUI/temp"
    
    try:
        if os.path.exists(temp_path):
            # Clear all files and subdirectories in the temp folder
            for filename in os.listdir(temp_path):
                file_path = os.path.join(temp_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
            
            print(f"Successfully cleared temp folder: {temp_path}")
        else:
            print(f"Temp folder does not exist: {temp_path}")
            
    except Exception as e:
        print(f"Error clearing temp folder {temp_path}: {e}")

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
        'illustrious-novaanimal': {'model_path': config["illustrious_novaanimal_model_path"]},
        'illustrious-kiwimix': {'model_path': config["illustrious_kiwimix_model_path"]},
        'sd3-medium': {'model_path': config["sd3_medium_model_path"]},
        'flux-unchained': {'model_path': config["flux_unchained_model_path"]},
        'wan': {'model_path': "a"},
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

def round_to_multiple_of_32(number):
    """Round a number to the nearest multiple of 32."""
    return round(number / 32) * 32

def calculate_img2video_upscale_size(base_size, upscale_ratio=1.5):
    """Calculate upscaled size for img2video, rounded to multiple of 32."""
    upscaled_size = base_size * upscale_ratio
    return round_to_multiple_of_32(upscaled_size)

def calculate_upscale_for_img2video(width, height, target_longest_side=1024):
    """Calculate scaling factor to ensure longest side is target_longest_side while maintaining aspect ratio."""
    current_longest_side = max(width, height)
    if current_longest_side >= target_longest_side:
        return width, height  # No upscaling needed
    
    scale_factor = target_longest_side / current_longest_side
    new_width = round_to_multiple_of_eight(width * scale_factor)
    new_height = round_to_multiple_of_eight(height * scale_factor)
    
    return new_width, new_height

app = Flask(__name__)
CORS(app)

@app.route('/get-lora-yaml', methods=['GET'])
def get_lora_yaml():
    return jsonify(lora_weights_map)

def add_watermark(image, text, data):
    try:
        if image is None:
            print("ERROR add_watermark: Input image is None")
            return None
            
        if not hasattr(image, 'width') or not hasattr(image, 'height'):
            print("ERROR add_watermark: Image missing width/height attributes")
            return None
        
        extras = data.get('extras', {})
        
        if extras.get("removeWatermark", False) == True:
            return image
        
        # create a font size based on the width and height of the image:
        data_height = image.height
        data_width = image.width
        
        if data_width <= 0 or data_height <= 0:
            print(f"ERROR add_watermark: Invalid image dimensions: {data_width}x{data_height}")
            return image  # Return original image instead of None
        
        text_image = Image.new('RGBA', (data_width, data_height), (255, 255, 255, 0))
        
        draw = ImageDraw.Draw(text_image)
        
        font_scale = data_height + data_width
        font_size = max(int(font_scale / 55), 12)  # Ensure minimum font size
            
        try:
            font = ImageFont.truetype("Amaranth-Regular.ttf", font_size)
        except (IOError, OSError) as e:
            print(f"Warning: Could not load custom font: {e}, using default")
            font = ImageFont.load_default()  # Load the default font in case of error

        # Use textbbox to get the bounding box of the text
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception as e:
            print(f"Warning: Could not get text bbox: {e}, using fallback")
            text_width = len(text) * font_size // 2
            text_height = font_size

        # Calculate the x, y position of the text
        x = max(0, image.width - 10 - text_width)  # Ensure x is not negative
        y = 10  # 10 pixels offset from the top edge

        outline_color = "black"
        outline_thickness = max(1, int(font_size / 20))  # Ensure minimum thickness
        
        # draw the outline using the outline thickness value:
        try:
            for i in range(1, outline_thickness + 1):
                draw.text((x - i, y - i), text, font=font, fill=outline_color)
                draw.text((x + i, y - i), text, font=font, fill=outline_color)
                draw.text((x - i, y + i), text, font=font, fill=outline_color)
                draw.text((x + i, y + i), text, font=font, fill=outline_color)

            # Text itself
            text_color = (255, 255, 255)
            draw.text((x, y), text, font=font, fill=text_color)
        except Exception as e:
            print(f"Warning: Could not draw text outline: {e}")
            # Just draw the text without outline
            try:
                draw.text((x, y), text, font=font, fill=(255, 255, 255))
            except Exception as e2:
                print(f"Error: Could not draw text at all: {e2}")
                return image  # Return original image if text drawing fails
        
        try:
            # change the text image to be lower opacity:
            text_image = text_image.convert('RGBA')
            text_image_data = text_image.getdata()
            
            new_text_image_data = []
            for item in text_image_data:
                if item[3] == 0:
                    new_text_image_data.append((255, 255, 255, 0))
                elif item[3] != 0 and item[2] != 255:
                    new_text_image_data.append((0, 0, 0, 170))
                elif item[3] != 0 and item[2] == 255:
                    new_text_image_data.append((255, 255, 255, 170))
                else:
                    new_text_image_data.append(item)
                    
            text_image.putdata(new_text_image_data)
            
            # merge the text image with the original image:
            image_rgba = image.convert('RGBA')
            final_image = Image.alpha_composite(image_rgba, text_image)
            
            # convert the image back to RGBA mode:
            final_image = final_image.convert('RGBA')
            
            return final_image
        except Exception as e:
            print(f"Error in watermark composition: {e}")
            return image  # Return original image if composition fails

    except Exception as e:
        print(f"Critical error in add_watermark: {e}")
        log_error(f"Critical error in add_watermark: {e}", data)
        return image  # Always return the original image rather than None

def process_webp_with_watermarks(webp_path, data):
    """
    Process a WebP file by splitting it into frames, adding watermarks to each frame,
    and then recreating the WebP. Returns a list containing the final WebP as a PIL Image.
    """
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: Processing WebP file: {webp_path} (attempt {attempt + 1}/{max_retries})")
            
            # Check if file exists before attempting to open
            if not os.path.exists(webp_path):
                print(f"ERROR: WebP file does not exist: {webp_path}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return None
            
            # Check file size to ensure it's not corrupted or empty
            file_size = os.path.getsize(webp_path)
            if file_size == 0:
                print(f"ERROR: WebP file is empty: {webp_path}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return None
            
            print(f"DEBUG: WebP file size: {file_size} bytes")
            
            # Open the original WebP with retry mechanism
            with Image.open(webp_path) as webp:
                print(f"DEBUG: Opened file format: {webp.format}")
                # Extract all frames from the WebP
                frames = []
                durations = []
                
                # Get WebP properties
                original_format = webp.format
                is_animated = getattr(webp, "is_animated", False)
                print(f"DEBUG: Is animated: {is_animated}, Original format: {original_format}")
                
                if not is_animated:
                    # If it's not animated, treat it as a single frame
                    frames = [webp.copy()]
                    durations = [100]  # Default duration
                    print(f"DEBUG: Single frame WebP detected")
                else:
                    # Extract all frames and their durations
                    frame_count = getattr(webp, "n_frames", 1)
                    print(f"DEBUG: Animated WebP with {frame_count} frames")
                    for i in range(frame_count):
                        webp.seek(i)
                        frame = webp.copy().convert('RGBA')
                        frames.append(frame)
                        
                        # Get frame duration (in milliseconds)
                        duration = webp.info.get('duration', 100)
                        durations.append(duration)
                
                # Upscale frames back to target size (1024px default, or upscaled if extras.upscale enabled)
                upscaled_frames = []
                if frames:
                    # Get target dimensions using the same function
                    sample_frame = frames[0]
                    
                    # Calculate target longest side based on upscaling settings
                    base_target = 1024
                    if data.get('extras', {}).get('upscale', False):
                        upscale_ratio = global_settings.get('img2video_upscale_ratio', 1.5)
                        target_longest_side = round_to_multiple_of_32(base_target * upscale_ratio)
                        print(f"WebP upscaling enabled: base {base_target} -> {target_longest_side} (ratio: {upscale_ratio})")
                    else:
                        target_longest_side = base_target
                        
                    target_width, target_height = calculate_upscale_for_img2video(sample_frame.width, sample_frame.height, target_longest_side=target_longest_side)
                    print(f"Upscaling WebP frames: {sample_frame.width}x{sample_frame.height} -> {target_width}x{target_height}")
                    
                    for frame in frames:
                        upscaled_frame = frame.resize((target_width, target_height), Image.LANCZOS)
                        upscaled_frames.append(upscaled_frame)
                else:
                    upscaled_frames = frames
                
                # Add watermark to each upscaled frame
                watermarked_frames = []
                watermark_text = "JSCammie.com"
                
                for frame in upscaled_frames:
                    watermarked_frame = add_watermark(frame, watermark_text, data)
                    watermarked_frames.append(watermarked_frame)
                
                # Create a new WebP from watermarked frames
                if len(watermarked_frames) > 1:
                    # Check if this is a doubleImages request that needs frame splitting
                    expected_length = global_settings.get('img2video_length', 81)  # Use global setting
                    is_double_images = data.get('extras', {}).get('doubleImages', False)
                    
                    if is_double_images and len(watermarked_frames) >= (expected_length * 2):
                        print(f"DEBUG: Detected doubleImages request with {len(watermarked_frames)} frames, splitting into 2 videos of {expected_length} frames each")
                        
                        # Split frames into two separate videos
                        video1_frames = watermarked_frames[:expected_length]
                        video2_frames = watermarked_frames[expected_length:expected_length*2]
                        video1_durations = durations[:expected_length] if len(durations) >= expected_length else [100] * expected_length
                        video2_durations = durations[expected_length:expected_length*2] if len(durations) >= expected_length*2 else [100] * expected_length
                        
                        # Create two separate WebP files
                        webp_results = []
                        for video_idx, (frames_set, durations_set) in enumerate([(video1_frames, video1_durations), (video2_frames, video2_durations)], 1):
                            output_path = webp_path.replace('.webp', f'_watermarked_video{video_idx}.webp')
                            
                            # Convert frames to consistent mode
                            final_frames = []
                            for frame in frames_set:
                                if frame.mode != 'RGBA':
                                    frame = frame.convert('RGBA')
                                final_frames.append(frame)
                            
                            # Save individual video
                            final_frames[0].save(
                                output_path,
                                save_all=True,
                                append_images=final_frames[1:],
                                duration=durations_set,
                                loop=0,
                                format='WEBP',
                                optimize=True
                            )
                            
                            # Load the saved WebP and set format
                            video_webp = Image.open(output_path)
                            video_webp.format = 'WEBP'
                            
                            # Set animation properties
                            if hasattr(video_webp, 'is_animated'):
                                print(f"DEBUG: Video {video_idx} WebP is_animated: {video_webp.is_animated}")
                            else:
                                try:
                                    video_webp.seek(1)
                                    video_webp.seek(0)
                                    video_webp.is_animated = True
                                    print(f"DEBUG: Video {video_idx} manually set is_animated to True")
                                except:
                                    video_webp.is_animated = False
                                    print(f"DEBUG: Video {video_idx} could not seek frames, setting is_animated to False")
                            
                            webp_results.append(video_webp)
                        
                        print(f"DEBUG: Returning {len(webp_results)} separate WebP videos")
                        return webp_results
                    else:
                        # Single video case (original behavior)
                        print(f"DEBUG: Processing as single video with {len(watermarked_frames)} frames")
                        output_path = webp_path.replace('.webp', '_watermarked.webp')
                    
                    # Convert all frames to the same mode for consistency
                    final_frames = []
                    for frame in watermarked_frames:
                        if frame.mode != 'RGBA':
                            frame = frame.convert('RGBA')
                        final_frames.append(frame)
                    
                    # Save the new WebP with original timing
                    final_frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=final_frames[1:],
                        duration=durations,
                        loop=0,  # Infinite loop
                        format='WEBP',
                        optimize=True
                    )
                    
                    # Return the watermarked WebP - we can't use copy() as it only copies one frame
                    # Instead, we'll read the file again to preserve the animation
                    final_webp = Image.open(output_path)
                    final_webp.format = 'WEBP'  # Preserve format information
                    
                    # Manually set animation properties for debugging
                    if hasattr(final_webp, 'is_animated'):
                        print(f"DEBUG: Final WebP is_animated: {final_webp.is_animated}")
                    else:
                        print(f"DEBUG: Final WebP missing is_animated attribute")
                        # Try to manually set animation properties
                        try:
                            final_webp.seek(1)  # Try to seek to second frame
                            final_webp.seek(0)  # Go back to first frame
                            final_webp.is_animated = True
                            print(f"DEBUG: Manually set is_animated to True")
                        except:
                            final_webp.is_animated = False
                            print(f"DEBUG: Could not seek frames, setting is_animated to False")
                    
                    print(f"DEBUG: Returning animated WebP with format: {final_webp.format}")
                    return [final_webp]
                else:
                    # Single frame, return as WebP image with format preserved
                    for frame in watermarked_frames:
                        frame.format = 'WEBP'  # Ensure format is preserved for single frames
                    print(f"DEBUG: Returning single frame WebP with format: {watermarked_frames[0].format if watermarked_frames else 'None'}")
                    return watermarked_frames
                
        except Exception as e:
            print(f"Error processing WebP with watermarks (attempt {attempt + 1}/{max_retries}): {e}")
            log_error(f"Error processing WebP with watermarks (attempt {attempt + 1}/{max_retries}): {e}", {"webp_path": webp_path, "data": str(data)})
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            # Final attempt failed
            print(f"CRITICAL: Failed to process WebP after {max_retries} attempts: {webp_path}")
            return None
    
    # This should never be reached, but just in case
    print(f"CRITICAL: Unexpected exit from WebP processing: {webp_path}")
    return None

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
        time.sleep(0.1)
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
            
    # if the request_type is img2video, then manually add 2 loras, not available in the lora_weights_map:
    if data['request_type'] == "img2video":
        # Check if lightning LoRA should be disabled
        disable_lightning = data.get('disableLightningLora', False)
        
        if not disable_lightning:
            # Get the video lora data from the YAML file
            style_wanlightning = lora_weights_map.get('style', {}).get('style-wanlightning')
            
            # Get lightning strength from request data, default to 1.0
            lightning_strength = float(data.get('lightningStrength', 1.0))
            print(f"DEBUG: Using lightning strength: {lightning_strength}")
            
            if style_wanlightning:
                # Use the same format as other LoRAs: (lora_path, strength)
                data['loras_data']['style-style-wanlightning'] = style_wanlightning['lora'], lightning_strength
                print(f"DEBUG: Added lightning LoRA with strength {lightning_strength}")
        else:
            print(f"DEBUG: Lightning LoRA disabled by user setting")
        
def add_metadata(image, metadata, data):
    try:
        if not metadata or not isinstance(metadata, dict):
            print("Warning: Invalid metadata provided")
            return None
            
        meta = PngImagePlugin.PngInfo()
       
        for key, value in metadata.items():
            try:
                # Ensure both key and value are strings and handle None values
                key_str = str(key) if key is not None else "unknown_key"
                value_str = str(value) if value is not None else "unknown_value"
                meta.add_text(key_str, value_str)
            except Exception as e:
                print(f"Warning: Could not add metadata key '{key}': {e}")
                continue

        return meta
    except Exception as e:
        print(f"Error creating metadata: {e}")
        log_error(f"Error creating metadata: {e}", {"metadata": str(metadata)})
        return None

def add_webp_metadata(metadata):
    """Create EXIF metadata for WebP files"""
    try:
        if not metadata or not isinstance(metadata, dict):
            print("Warning: Invalid metadata provided for WebP")
            return None
            
        from PIL.ExifTags import TAGS
        from PIL import ExifTags
        import json
        
        # Create a JSON string containing all metadata for the UserComment field
        metadata_json = json.dumps(metadata, indent=None, separators=(',', ':'))
        
        # Create basic EXIF data
        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "1st": {},
            "thumbnail": None
        }
        
        # Add metadata as UserComment (tag 37510 / 0x9286)
        exif_dict["Exif"][37510] = metadata_json.encode('utf-8')  # UserComment
        
        # Add some basic info to EXIF if available
        if "model" in metadata:
            exif_dict["0th"][272] = str(metadata["model"])  # Model
        if "prompt" in metadata:
            exif_dict["0th"][270] = str(metadata["prompt"])[:100]  # ImageDescription (truncated)
        if "generation_date" in metadata:
            exif_dict["0th"][306] = str(metadata["generation_date"])  # DateTime
            
        try:
            import piexif
            exif_bytes = piexif.dump(exif_dict)
            return exif_bytes
        except ImportError:
            print("Warning: piexif not available, WebP metadata will be limited")
            return None
        except Exception as e:
            print(f"Warning: Could not create EXIF data: {e}")
            return None
            
    except Exception as e:
        print(f"Error creating WebP metadata: {e}")
        log_error(f"Error creating WebP metadata: {e}", {"metadata": str(metadata)})
        return None


def save_image(request_id, output_image, model_type, data):
    try:
        # Validate input parameters
        if output_image is None:
            print("ERROR save_image: output_image is None")
            return None
            
        if not hasattr(output_image, 'width') or not hasattr(output_image, 'height'):
            print("ERROR save_image: output_image missing width/height attributes")
            return None
            
        if output_image.width <= 0 or output_image.height <= 0:
            print(f"ERROR save_image: Invalid image dimensions: {output_image.width}x{output_image.height}")
            return None
        
        # Generate metadata
        metadata = {
            "request_id": request_id,
            "model_type": model_type,
            "prompt": data.get('true_prompt', ''),
            "negative_prompt": str(data.get('negative_prompt', '')),
            "loras": data.get('lora', []),
            "lora_strengths": data.get('lora_strengths', []),
            "steps": data.get('steps', 20),
            "seed": data.get('seedNumber', -1),
            "CFG": data.get('guidance', 7.0),
            "model": data.get('og_model', ''),
            "generation_date": datetime.datetime.now().isoformat(),
            "accountId": str(data.get('accountId', '0')),
            "scheduler": data.get('scheduler', 'eulera')
        }

        # Check if this is a WebP (img2video) - in this case, watermarks are already applied
        print(f"DEBUG save_image: model_type={model_type}, hasattr format={hasattr(output_image, 'format')}")
        if hasattr(output_image, 'format'):
            print(f"DEBUG save_image: output_image.format={output_image.format}")
        
        # Additional WebP detection
        if hasattr(output_image, 'is_animated'):
            print(f"DEBUG save_image: output_image.is_animated={output_image.is_animated}")
        else:
            print(f"DEBUG save_image: output_image has no is_animated attribute")
            
        # Check all conditions for WebP processing
        is_img2video = model_type == "img2video"
        has_format_attr = hasattr(output_image, 'format')
        is_webp_format = has_format_attr and output_image.format == 'WEBP'
        
        print(f"DEBUG save_image: is_img2video={is_img2video}, has_format_attr={has_format_attr}, is_webp_format={is_webp_format}")
        
        if model_type == "img2video" and hasattr(output_image, 'format') and output_image.format == 'WEBP':
            print(f"DEBUG save_image: Processing as WebP img2video")
            try:
                # Create WebP metadata
                webp_metadata = add_webp_metadata(metadata)
                if webp_metadata:
                    print(f"DEBUG save_image: Created WebP EXIF metadata")
                else:
                    print(f"DEBUG save_image: No WebP metadata created")
                
                # For WebPs, save as WebP format and encode to base64
                buffered = io.BytesIO()
                # Save with save_all=True to preserve all frames for animated WebPs
                save_kwargs = {"format": "WEBP", "optimize": True}
                if webp_metadata:
                    save_kwargs["exif"] = webp_metadata
                    
                if hasattr(output_image, 'is_animated') and output_image.is_animated:
                    print(f"DEBUG save_image: Saving as animated WebP with metadata")
                    save_kwargs["save_all"] = True
                    output_image.save(buffered, **save_kwargs)
                else:
                    print(f"DEBUG save_image: Saving as single frame WebP with metadata")
                    output_image.save(buffered, **save_kwargs)
                
                # Validate buffered data
                buffered_data = buffered.getvalue()
                if len(buffered_data) == 0:
                    print("ERROR save_image: WebP buffer is empty")
                    return None
                    
                img_str = base64.b64encode(buffered_data).decode()
                if not img_str:
                    print("ERROR save_image: WebP base64 encoding failed")
                    return None
                
                # Always save as output.webp for testing purposes (enabled by default for img2video)
                try:
                    save_kwargs = {"optimize": True}
                    if webp_metadata:
                        save_kwargs["exif"] = webp_metadata
                        
                    if hasattr(output_image, 'is_animated') and output_image.is_animated:
                        save_kwargs["save_all"] = True
                        output_image.save("output.webp", **save_kwargs)
                    else:
                        output_image.save("output.webp", **save_kwargs)
                    print(f"Saved img2video output as output.webp with metadata for testing")
                except Exception as e:
                    print(f"Warning: Could not save output.webp: {e}")
                
                if data.get('save_image', False) == True:
                    try:
                        # create dir if you need to:
                        os.makedirs("./output_images", exist_ok=True)
                        
                        save_kwargs = {"optimize": True}
                        if webp_metadata:
                            save_kwargs["exif"] = webp_metadata
                            
                        if hasattr(output_image, 'is_animated') and output_image.is_animated:
                            save_kwargs["save_all"] = True
                            output_image.save(f"./output_images/{request_id}.webp", **save_kwargs)
                        else:
                            output_image.save(f"./output_images/{request_id}.webp", **save_kwargs)
                        print(f"Saved img2video WebP with metadata to output_images/{request_id}.webp")
                    except Exception as e:
                        print(f"Warning: Could not save to output_images: {e}")
                    
                print(f"DEBUG save_image: Returning WebP format response")
                return {
                    "width": output_image.width,
                    "height": output_image.height,
                    "base64": img_str,
                    "format": "webp"
                }
            except Exception as e:
                print(f"ERROR save_image: WebP processing failed: {e}")
                log_error(f"WebP processing failed in save_image: {e}", {"request_id": request_id, "model_type": model_type})
                return None
        else:
            print(f"DEBUG save_image: Processing as PNG (not WebP img2video)")
            try:
                # Regular image processing with watermarks
                watermark_text = "JSCammie.com"
                watermarked_image = add_watermark(output_image, watermark_text, data)
                
                if watermarked_image is None:
                    print("ERROR save_image: Watermarking failed")
                    return None
                
                meta = add_metadata(watermarked_image, metadata, data)
                if meta is None:
                    print("WARNING save_image: Metadata creation failed, proceeding without metadata")
                    meta = None
                
                buffered = io.BytesIO()
                if meta:
                    watermarked_image.save(buffered, format="PNG", pnginfo=meta)
                else:
                    watermarked_image.save(buffered, format="PNG")
                
                # Validate buffered data
                buffered_data = buffered.getvalue()
                if len(buffered_data) == 0:
                    print("ERROR save_image: PNG buffer is empty")
                    return None
                    
                img_str = base64.b64encode(buffered_data).decode()
                if not img_str:
                    print("ERROR save_image: PNG base64 encoding failed")
                    return None
                    
                if data.get('save_image', False) == True:
                    try:
                        # create dir if you need to:
                        os.makedirs("./output_images", exist_ok=True)
                        if meta:
                            watermarked_image.save(f"./output_images/{request_id}.png", pnginfo=meta)
                        else:
                            watermarked_image.save(f"./output_images/{request_id}.png")
                    except Exception as e:
                        print(f"Warning: Could not save to output_images: {e}")

                return {
                    "width": output_image.width,
                    "height": output_image.height,
                    "base64": img_str,
                    "format": "png"
                }
            except Exception as e:
                print(f"ERROR save_image: PNG processing failed: {e}")
                log_error(f"PNG processing failed in save_image: {e}", {"request_id": request_id, "model_type": model_type})
                return None
            
    except Exception as e:
        print(f"CRITICAL ERROR save_image: Unexpected error: {e}")
        log_error(f"Critical error in save_image: {e}", {"request_id": request_id, "model_type": model_type})
        return None

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# create websocket client:
import websocket

def check_for_image(prompt_id, data):
    previous_comfyui_response = None
    max_wait_time = 300  # Maximum wait time in seconds (5 minutes)
    start_time = time.time()
    
    def print_response_if_changed(message, comfyui_response):
        nonlocal previous_comfyui_response
        if comfyui_response != previous_comfyui_response:
            print(message, comfyui_response)
            previous_comfyui_response = comfyui_response
    
    def safe_file_access(file_path, max_retries=3):
        """Safely access a file with retry mechanism"""
        for attempt in range(max_retries):
            try:
                if not os.path.exists(file_path):
                    print(f"File does not exist: {file_path} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return None
                
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    print(f"File is empty: {file_path} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return None
                
                with Image.open(file_path) as img:
                    img_copy = img.copy()
                    img_copy.format = img.format
                    print(f"Successfully loaded image: {file_path} ({file_size} bytes)")
                    return img_copy
                    
            except Exception as e:
                print(f"Error loading image {file_path} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None
        return None
    
    while True:
        # Check for timeout
        if time.time() - start_time > max_wait_time:
            print(f"TIMEOUT: Generation took longer than {max_wait_time} seconds")
            log_error(f"Generation timeout for prompt_id {prompt_id}", data)
            return None
            
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

                # Handle different output nodes based on request type
                processed_files = []  # Track files we process for cleanup
                
                if data['request_type'] == "img2video":
                    # For img2video, node 321 outputs WebP files
                    # Try different possible output keys for video nodes
                    node_321_output = image_data.get('321', {})
                    print("Node 321 full output: ", node_321_output)
                    
                    # VHS_VideoCombine can output under 'gifs', 'videos', or 'filenames' (note: 'gifs' key name unchanged for compatibility)
                    webp_list = (node_321_output.get('gifs', []) or 
                               node_321_output.get('videos', []) or 
                               node_321_output.get('filenames', []))
                    
                    print("Image Info 321 (WebP): ", webp_list)
                    
                    for webp_info in webp_list:
                        webp_filename = webp_info.get('filename') if isinstance(webp_info, dict) else webp_info
                        if webp_filename:
                            processed_files.append(webp_filename)  # Track this file
                            webp_path = os.path.join(data['images_dir'], webp_filename)
                            print(f"DEBUG: Attempting to process WebP: {webp_path}")
                            
                            # Check if file exists before processing
                            if not os.path.exists(webp_path):
                                print(f"ERROR: WebP file not found: {webp_path}")
                                log_error(f"WebP file not found: {webp_path}", data)
                                continue
                            
                            try:
                                # Split WebP into frames, add watermarks, and recreate WebP
                                watermarked_webp_frames = process_webp_with_watermarks(webp_path, data)
                                if watermarked_webp_frames and len(watermarked_webp_frames) > 0:
                                    # Validate each frame before adding
                                    valid_frames = []
                                    for frame in watermarked_webp_frames:
                                        if frame is not None and hasattr(frame, 'width') and hasattr(frame, 'height'):
                                            valid_frames.append(frame)
                                        else:
                                            print(f"WARNING: Invalid frame detected in WebP processing")
                                    
                                    if valid_frames:
                                        output_images.extend(valid_frames)
                                        print(f"DEBUG check_for_image: Added {len(valid_frames)} valid WebP frames to output_images")
                                        for i, frame in enumerate(valid_frames):
                                            print(f"DEBUG check_for_image: Frame {i} format: {getattr(frame, 'format', 'No format attr')}")
                                    else:
                                        print(f"ERROR: No valid frames found in processed WebP")
                                        log_error(f"No valid frames in processed WebP: {webp_path}", data)
                                else:
                                    print(f"ERROR: WebP processing returned no frames: {webp_path}")
                                    log_error(f"WebP processing returned no frames: {webp_path}", data)
                                print(f"Processed WebP {webp_path}")
                            except Exception as webp_error:
                                print(f"CRITICAL: Failed to process WebP {webp_path}: {webp_error}")
                                log_error(f"Failed to process WebP {webp_path}: {webp_error}", data)
                                # Don't continue to next iteration, this is a critical failure
                                continue
                    
                    # For img2video, ONLY process WebP from node 321, ignore any PNG from node 64
                    print(f"DEBUG check_for_image: img2video completed, processed {len(output_images)} WebP images, ignoring any PNG outputs")
                else:
                    # For other request types, node 64 outputs regular images
                    image_info_64 = image_data.get('64', {}).get('images', [])
                    print("Image Info 64: ", image_info_64)

                    for image_info in image_info_64:
                        image_filename = image_info.get('filename')
                        if image_filename:
                            processed_files.append(image_filename)  # Track this file
                            image_path = os.path.join(data['images_dir'], image_filename)
                            print(f"DEBUG: Attempting to load PNG: {image_path}")
                            
                            # Use safe file access with retry mechanism
                            img_copy = safe_file_access(image_path)
                            if img_copy is not None:
                                # Validate the image
                                if hasattr(img_copy, 'width') and hasattr(img_copy, 'height') and img_copy.width > 0 and img_copy.height > 0:
                                    output_images.append(img_copy)
                                    print(f"Successfully loaded and validated image {image_path}")
                                else:
                                    print(f"ERROR: Invalid image dimensions: {image_path}")
                                    log_error(f"Invalid image dimensions: {image_path}", data)
                            else:
                                print(f"CRITICAL: Failed to load image after retries: {image_path}")
                                log_error(f"Failed to load image after retries: {image_path}", data)
                
                # Comprehensive validation of output_images
                if output_images and len(output_images) > 0:
                    # Validate each image in the list
                    valid_images = []
                    for i, img in enumerate(output_images):
                        if img is not None and hasattr(img, 'width') and hasattr(img, 'height'):
                            if img.width > 0 and img.height > 0:
                                valid_images.append(img)
                                print(f"Image {i} validated: {img.width}x{img.height}")
                            else:
                                print(f"WARNING: Image {i} has invalid dimensions: {img.width}x{img.height}")
                        else:
                            print(f"WARNING: Image {i} is None or missing dimensions")
                    
                    if valid_images:
                        print(f"Images fetched successfully. Valid images: {len(valid_images)}/{len(output_images)}")
                        print(f"Processed files to cleanup: {processed_files}")
                        return {"images": valid_images, "processed_files": processed_files}
                    else:
                        print(f"ERROR: No valid images found. Total processed: {len(output_images)}")
                        log_error(f"No valid images found in output_images. Total: {len(output_images)}", data)
                        return None
                else:
                    print("ERROR: No images were loaded from ComfyUI output.")
                    log_error("No images loaded from ComfyUI output", {
                        "prompt_id": prompt_id,
                        "request_type": data.get('request_type', 'unknown'),
                        "images_dir": data.get('images_dir', 'unknown'),
                        "status_info": status_info
                    })
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
        
        result = check_for_image(prompt_id, data)
        
        # Handle new return format from check_for_image
        if result is None:
            error_msg = "Generation completed but no images were found. This may be due to ComfyUI processing issues or file access problems."
            print(f"CRITICAL: {error_msg}")
            log_error(f"No output images for request {request_id}", {
                "prompt_id": prompt_id,
                "request_type": data.get('request_type', 'unknown'),
                "model": data.get('model', 'unknown'),
                "images_dir": data.get('images_dir', 'unknown')
            })
            results[request_id] = {"status": "error", "message": error_msg}
            queue_item.status = "error"
            return "skipped"
        
        output_images = result["images"]
        processed_files = result["processed_files"]
        
        if len(output_images) == 0:
            error_msg = "Generation completed but the output images list is empty."
            print(f"CRITICAL: {error_msg}")
            log_error(f"Empty output images list for request {request_id}", data)
            results[request_id] = {"status": "error", "message": error_msg}
            queue_item.status = "error"
            return "skipped"
        
        # Validate each image in output_images
        valid_images = []
        for i, img in enumerate(output_images):
            if img is not None and hasattr(img, 'width') and hasattr(img, 'height'):
                if img.width > 0 and img.height > 0:
                    valid_images.append(img)
                else:
                    print(f"WARNING: Image {i} has invalid dimensions: {img.width}x{img.height}")
            else:
                print(f"WARNING: Image {i} is None or corrupted")
        
        if len(valid_images) == 0:
            error_msg = "Generation completed but all output images are corrupted or invalid."
            print(f"CRITICAL: {error_msg}")
            log_error(f"All output images invalid for request {request_id}", {
                "total_images": len(output_images),
                "valid_images": len(valid_images)
            })
            results[request_id] = {"status": "error", "message": error_msg}
            queue_item.status = "error"
            return "skipped"
        
        # Use only valid images
        output_images = valid_images
        print(f"Using {len(valid_images)} valid images out of {len(output_images)} total")
        
        # Note: File cleanup will happen after image processing is complete
        
        

        # Watermark all images:
        # Assume save_image is defined correctly to handle the image processing and saving
        
        Base64Images = []
        for i in range(len(output_images)):
            img = output_images[i]
            print(f"DEBUG process_request: Processing image {i}, format: {getattr(img, 'format', 'No format attr')}, request_type: {data['request_type']}")
            
            try:
                # Validate image before processing
                if img is None:
                    print(f"ERROR: Image {i} is None")
                    continue
                    
                if not hasattr(img, 'width') or not hasattr(img, 'height'):
                    print(f"ERROR: Image {i} missing width/height attributes")
                    continue
                    
                if img.width <= 0 or img.height <= 0:
                    print(f"ERROR: Image {i} has invalid dimensions: {img.width}x{img.height}")
                    continue
                
                watermarked_image = save_image(request_id, img, data['request_type'], data)
                
                if watermarked_image is None:
                    print(f"ERROR: save_image returned None for image {i}")
                    log_error(f"save_image returned None for image {i}", {"request_id": request_id, "request_type": data['request_type']})
                    continue
                    
                if not isinstance(watermarked_image, dict) or 'base64' not in watermarked_image:
                    print(f"ERROR: save_image returned invalid format for image {i}")
                    log_error(f"save_image returned invalid format for image {i}", {"watermarked_image": str(watermarked_image)})
                    continue
                
                print(f"DEBUG process_request: save_image returned format: {watermarked_image.get('format', 'No format field')}")
                Base64Images.append(watermarked_image)
                
            except Exception as e:
                print(f"ERROR: Failed to process image {i}: {e}")
                log_error(f"Failed to process image {i}: {e}", {"request_id": request_id, "image_info": str(img)})
                continue
            
        PIL_Images = []
        
        # convert the base64 images to PIL images:
        for i in range(len(Base64Images)):
            img_str = Base64Images[i]['base64']
            img_data = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_data))
            
            # For WebPs, use the first frame for hash calculation
            if hasattr(img, 'is_animated') and img.is_animated:
                img.seek(0)  # Go to first frame
                img = img.copy()  # Copy the first frame as a static image
            
            PIL_Images.append(img)
        
        # Final validation: ensure we have valid Base64Images
        if not Base64Images or len(Base64Images) == 0:
            error_msg = "Failed to process any images to base64 format."
            print(f"CRITICAL: {error_msg}")
            log_error(f"No Base64Images generated for request {request_id}", {
                "output_images_count": len(output_images),
                "processing_errors": "Check logs above for individual image errors"
            })
            results[request_id] = {"status": "error", "message": error_msg}
            queue_item.status = "error"
            return "skipped"
            
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

        results[request_id] = {
            "images": Base64Images,
            "additionalInfo": {
                "seed": data['seed'],
                "executiontime": time.time() - start_time,
                "totaltime": total_time,
                "timestamp": time.time()
            },
            "historyData": {
                "account_id": data['accountId'],
                "prompt": data['true_prompt'],
                "negative_prompt": data['true_negative_prompt'],
                "model": data['og_model'],
                "aspect_ratio": data['aspect_ratio'],
                "loras": data['lora'],
                "lora_strengths": data['lora_strengths'],
                "steps": data['steps'],
                "cfg": data['guidance'],
                "seed": data['seed'],
            },
            "fastqueue": data['fastqueue'],
            "creditsRequired": str(data['creditsRequired']),
        }
        
        # Debug: Check what format is being stored
        print(f"DEBUG process_request: Stored {len(Base64Images)} images in results")
        for i, img_data in enumerate(Base64Images):
            if img_data:
                print(f"DEBUG process_request: Result image {i} format: {img_data.get('format', 'No format field')}")
        
        # Clean up processed files and temp folder after successful processing
        try:
            # Clear specific files that were processed from output folder
            if processed_files:
                clear_processed_files(processed_files, data['images_dir'])
            
            # Clear temp folder completely
            clear_temp_folder()
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        
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
        time.sleep(0.1)  # Short sleep to prevent CPU overutilization
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
            new_global_settings = yaml.safe_load(settings_content)
            
            # Check if EasyCache settings have changed (only if global_settings is already initialized)
            easycache_settings_changed = False
            if 'global_settings' in globals() and global_settings is not None:
                easycache_keys = ['easycache_enabled', 'easycache_img2video_enabled', 'easycache_reuse_threshold', 'easycache_start_percent', 'easycache_end_percent', 'easycache_verbose']
                for key in easycache_keys:
                    if global_settings.get(key) != new_global_settings.get(key):
                        easycache_settings_changed = True
                        break
            
            global_settings = new_global_settings
            
            # Update EasyCache settings in queued requests if settings changed
            if easycache_settings_changed:
                update_easycache_in_queues()

        except Exception as e:
            print(f"Error reading the settings yaml file: {e}")
        time.sleep(1)

def update_easycache_in_queues():
    """Update LazyCache node settings in all queued requests"""
    try:
        # Update requests in queue 0
        for queue_item in request_queue_0:
            if queue_item.status == "queued" and 'request_json' in queue_item.data:
                update_easycache_node_in_request(queue_item.data['request_json'])
        
        # Update requests in queue 1  
        for queue_item in request_queue_1:
            if queue_item.status == "queued" and 'request_json' in queue_item.data:
                update_easycache_node_in_request(queue_item.data['request_json'])
                
        print("DEBUG: Updated LazyCache settings in all queued requests")
    except Exception as e:
        print(f"Error updating LazyCache in queues: {e}")

def update_easycache_node_in_request(request_json):
    """Update LazyCache node settings in a specific request JSON"""
    try:
        # Find EasyCache node in the request
        easycache_node_id = None
        for key in request_json:
            if (key.isnumeric() and 
                request_json[key].get('class_type') == 'LazyCache'):
                easycache_node_id = key
                break
        
        if easycache_node_id:
            # Update the EasyCache node inputs with current global settings
            request_json[easycache_node_id]['inputs'].update({
                'reuse_threshold': global_settings.get('easycache_reuse_threshold', 0.2),
                'start_percent': global_settings.get('easycache_start_percent', 0.15),
                'end_percent': global_settings.get('easycache_end_percent', 0.95),
                'verbose': global_settings.get('easycache_verbose', False)
            })
            print(f"DEBUG: Updated LazyCache node {easycache_node_id} settings")
    except Exception as e:
        print(f"Error updating LazyCache node: {e}")
    
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
    import time
    
    # First attempt to get result
    result = results.get(request_id)
    print(f"DEBUG get_result: Request ID {request_id}, result exists: {result is not None}")
    
    # If no result found, wait a small delay and check again
    # This allows for temporary storage of generations while they're being processed
    if not result:
        print(f"DEBUG get_result: No result found for {request_id}, waiting 500ms and checking again...")
        time.sleep(0.5)  # 500ms delay
        result = results.get(request_id)
        print(f"DEBUG get_result: After retry, result exists: {result is not None}")
    
    # Validate result before processing
    if result:
        # Check if this is an error result
        if result.get('status') == 'error':
            print(f"DEBUG get_result: Error result for {request_id}: {result.get('message', 'Unknown error')}")
            results.pop(request_id, None)  # Clean up
            return jsonify(result)
        
        # Validate images exist in result
        if 'images' not in result:
            print(f"ERROR get_result: No images key in result for {request_id}")
            results.pop(request_id, None)
            return jsonify({"status": "error", "message": "Result contains no images data"})
        
        images = result.get('images', [])
        if not images or len(images) == 0:
            print(f"ERROR get_result: Empty images list for {request_id}")
            results.pop(request_id, None)
            return jsonify({"status": "error", "message": "Result contains no images"})
        
        print(f"DEBUG get_result: Found result with {len(images)} images")
        
        # Check if we need to convert PIL images to base64:
        if images and hasattr(images[0], 'format'):
            print(f"DEBUG get_result: Converting PIL images to base64")
            converted_images = []
            
            for i in range(len(images)):
                img = images[i]
                
                try:
                    # Validate PIL image
                    if img is None:
                        print(f"ERROR get_result: Image {i} is None")
                        continue
                        
                    if not hasattr(img, 'width') or not hasattr(img, 'height'):
                        print(f"ERROR get_result: Image {i} missing dimensions")
                        continue
                        
                    if img.width <= 0 or img.height <= 0:
                        print(f"ERROR get_result: Image {i} has invalid dimensions: {img.width}x{img.height}")
                        continue
                    
                    buffered = io.BytesIO()
                    
                    # Handle different image formats
                    if hasattr(img, 'format') and img.format == 'WEBP':
                        print(f"DEBUG get_result: Image {i} is WebP format")
                        # For animated WebPs, preserve all frames
                        if hasattr(img, 'is_animated') and img.is_animated:
                            img.save(buffered, format="WEBP", save_all=True, optimize=True)
                        else:
                            img.save(buffered, format="WEBP", optimize=True)
                        img_format = "webp"
                    else:
                        print(f"DEBUG get_result: Image {i} is not WebP, using PNG")
                        img.save(buffered, format="PNG")
                        img_format = "png"
                    
                    # Validate the saved data
                    buffered_data = buffered.getvalue()
                    if len(buffered_data) == 0:
                        print(f"ERROR get_result: Image {i} produced empty buffer")
                        continue
                        
                    img_str = base64.b64encode(buffered_data).decode()
                    if not img_str:
                        print(f"ERROR get_result: Image {i} failed base64 encoding")
                        continue
                    
                    converted_images.append({
                        "base64": img_str, 
                        "width": img.width, 
                        "height": img.height,
                        "format": img_format
                    })
                    print(f"DEBUG get_result: Successfully converted image {i} format: {img_format}")
                    
                except Exception as e:
                    print(f"ERROR get_result: Failed to convert image {i}: {e}")
                    log_error(f"Failed to convert PIL image {i} to base64: {e}", {"request_id": request_id})
                    continue
            
            # Update result with converted images
            if converted_images:
                result['images'] = converted_images
                print(f"DEBUG get_result: Successfully converted {len(converted_images)}/{len(images)} images")
            else:
                print(f"ERROR get_result: Failed to convert any images for {request_id}")
                results.pop(request_id, None)
                return jsonify({"status": "error", "message": "Failed to convert images to proper format"})
                
        else:
            # Images are already in base64 format - validate them
            print(f"DEBUG get_result: Images already in base64 format")
            valid_images = []
            
            for i, img_data in enumerate(images):
                if img_data and isinstance(img_data, dict):
                    if 'base64' in img_data and img_data['base64']:
                        valid_images.append(img_data)
                        print(f"DEBUG get_result: Image {i} format: {img_data.get('format', 'No format field')}")
                    else:
                        print(f"ERROR get_result: Image {i} missing or empty base64 data")
                else:
                    print(f"ERROR get_result: Image {i} is not a valid dict or is None")
            
            if not valid_images:
                print(f"ERROR get_result: No valid base64 images found for {request_id}")
                results.pop(request_id, None)
                return jsonify({"status": "error", "message": "No valid image data found"})
            
            result['images'] = valid_images
    
    # remove the result from the results dictionary:
    results.pop(request_id, None)
    
    if result:
        # Final validation before returning
        if 'images' in result and result['images'] and len(result['images']) > 0:
            return jsonify(result)
        else:
            print(f"ERROR get_result: Final validation failed - no valid images for {request_id}")
            return jsonify({"status": "error", "message": "Final validation failed - no valid images found"})
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



@app.route('/get-queue-numbers', methods=['GET'])
def get_queue_numbers():
    return jsonify({"queue_0": len(request_queue_0), "queue_1": len(request_queue_1)})


                
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        
        if global_settings.get('maintenance', False):
            return generate_error_response("Maintenance Mode is currently enabled, the requests that are already in the queue are being completed, please wait a minute or two and try again.", 503)

        data = request.json

        # print("Headers:", request.headers)
        # print("Data:", request.get_json())
        
        # DEBUG: Print model value to see what's actually being sent
        print(f"DEBUG: Received model value: '{data['model']}'")
                
        data['og_model'] = data['model']
        
        # Safety measures for wan model requests
        if data['model'] == "wan" or data['model'].startswith("wan"):
            # Check if this is NOT an img2video request - wan model only supports img2video
            if data.get('request_type') != "img2video":
                return generate_error_response("The Wan model is only available for Image-to-Video generation. Please refresh the page and select Image-to-Video mode to use this model.", 400)
            
            # Check if no image is provided for img2video with wan model
            if data.get('image') is None and data.get('input_image') is None:
                return generate_error_response("Wan model requires an input image for video generation. Please upload an image and try again.", 400)
                
        # Safety check for img2video requests without proper request type
        if (data.get('image') is not None or data.get('input_image') is not None) and data.get('request_type') not in ["img2img", "inpainting", "img2video"]:
            # User uploaded an image but request type is wrong - likely outdated frontend
            return generate_error_response("It looks like you're using an outdated version of the website. Please refresh the page (Ctrl+F5 or Cmd+Shift+R) and try again.", 400)
        
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
    
                    
        if data['model'].startswith("pdxl-"):
            if data['steps'] > global_settings['pdxl_max_steps']:
                data['steps'] = global_settings['pdxl_max_steps']
                
        if data['model'].startswith("illustrious-"):
            if data['steps'] > global_settings['illustrious_max_steps']:
                data['steps'] = global_settings['illustrious_max_steps']

        if data['request_type'] == "img2video":
            
            # enforce scheduler to be "lcm", "normal", or "ddim_video" for img2video
            if data.get('scheduler') not in ["lcm", "normal", "ddim_video"]:
                data['scheduler'] = "normal"
                print(f"Img2video scheduler enforced to 'normal' (was: {data.get('scheduler', 'undefined')})")
            
            # img2video requires an input image
            if data.get('image') is None and data.get('input_image') is None:
                return generate_error_response("img2video requires an input image. Please upload an image and try again.", 400)
                
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
        elif data['model'].startswith("wan-"):
            if len(data['lora']) > global_settings['max_loras_wan']:
                return generate_error_response(f"Maximum number of Lora options for Wan model is: {global_settings['max_loras_wan']}", 400)
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
            return generate_error_response("Your prompt contains words that are not allowed, please remove them and try again.", 400)

        data['accountId'] = data.get('accountId', "0")

        if data['accountId'] == "":
            data['accountId'] = "0"

        data['accountId'] = str(data['accountId'])

        data['prompt'] = randomize_string(data['prompt'])

        if str(data['prompt']) == "{'status': 'error', 'message': 'Mismatched brackets'}":
            return generate_error_response("Mismatched brackets ('{}' brackets are used to denote a random choice, and must be used in pairs, here is an example of a correct usage: '{woman|man} with {long|short} hair')", 400)

        if data['steps'] > 126:
            return generate_error_response("You have reached the limit of 125 steps per request. Please reduce the number of steps and try again.", 400)

        if data['quantity'] > 5:
            return generate_error_response("You have reached the limit of 4 images per request. Please reduce the number of images and try again.", 400)

        if len(data['lora']) > 5:
            return generate_error_response("You have reached the limit of 5 Lora options. Please deselect some and try again.", 400)

        if data['seed'] is None:
            data['seed'] = -1

        if int(data['seed']) > 2**32 - 1:
            return generate_error_response("Seed is too large. Please use a seed between -1 and 4294967295.", 400)

        if int(data['seed']) < -1:
            return generate_error_response("Seed is too small. Please use a seed between -1 and 4294967295.", 400)
    

        model = data.get("model", "sonic")
        
        if model == None:
            return generate_error_response("Model is not specified, please try re-selecting the model and try again.", 400)
        
        data['true_negative_prompt'] = data.get("negativeprompt", "")

        # if data.get("model", "sonic").startswith("pdxl-") or data.get("model", "sonic").startswith("flux-") or data.get("model", "sonic").startswith("illustrious-"):
            
        #     negative_embedding_words_pdxl = ""
        #     negative_prompt_final = negative_embedding_words_pdxl + data.get("negativeprompt", "")
            
        # else:
            
        #     negative_embedding_words_sd15 = "boring_e621_v4, fcNeg, fluffynegative, badyiffymix41, gnarlysick-neg, negative_hand-neg, "
        #     negative_prompt_final = negative_embedding_words_sd15 + data.get("negativeprompt", "")
                    
        if data['model'].startswith("pdxl-") or data['model'].startswith("flux-") or data['model'].startswith("illustrious-") or data['model'].startswith("wan"):
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
                
                # Special handling for img2video: upscale so longest side is 1024 (or higher if extras.upscale enabled)
                if data['request_type'] == "img2video":
                    # Calculate target longest side based on upscaling settings
                    base_target = 1024
                    if data.get('extras', {}).get('upscale', False):
                        upscale_ratio = global_settings.get('img2video_upscale_ratio', 1.5)
                        target_longest_side = round_to_multiple_of_32(base_target * upscale_ratio)
                        print(f"Img2video output upscaling enabled: base {base_target} -> {target_longest_side} (ratio: {upscale_ratio})")
                    else:
                        target_longest_side = base_target
                        
                    new_width, new_height = calculate_upscale_for_img2video(data['image'].width, data['image'].height, target_longest_side=target_longest_side)
                    print(f"Img2video upscaling: {data['image'].width}x{data['image'].height} -> {new_width}x{new_height}")
                else:
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
            
            
        if data.get('fastqueue', False) == True:
            data['gpu_id'] = 1
        else:
            data['gpu_id'] = 0
            
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
            "gpu_id": data.get("gpu_id", 0),
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
            
        # if any extras are enabled and the request_type is img2video, then return an error, apart from remove watermark, upscale, and doubleImages, allow those:
        disallowed_extras = {k: v for k, v in validated_data['extras'].items() if k not in ['removeWatermark', 'upscale', 'doubleImages'] and v == True}
        if disallowed_extras and validated_data['request_type'] == "img2video":
            return generate_error_response(f"Only removeWatermark, upscale, and doubleImages extras are supported for {validated_data['request_type']}", 400)
        
        # if regionalPromptSettings is true with a request_type of img2video, then return an error:
        if validated_data['regionalPromptSettings']['status'] == "true" and validated_data['request_type'] == "img2video":
            return generate_error_response("Regional Prompt Settings are not supported for img2video", 400)
        
            
            
            
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
            
        
        # Update validated_data with properly rounded dimensions  
        validated_data['width'] = round_to_multiple_of_eight(validated_data['width'])
        validated_data['height'] = round_to_multiple_of_eight(validated_data['height'])
        
        # For img2img/inpainting with upscaling enabled, resize the input image to match upscaled dimensions
        if validated_data.get('extras', {}).get('upscale', False) and validated_data.get('request_type') in ['img2img', 'inpainting']:
            upscale_scale = validated_data.get('upscale_scale', 1.0)
            original_width = validated_data['width']
            original_height = validated_data['height']
            new_upscaled_width = round_to_multiple_of_eight(int(validated_data['width'] * upscale_scale))
            new_upscaled_height = round_to_multiple_of_eight(int(validated_data['height'] * upscale_scale))
            
            # Resize the input image to match upscaled dimensions
            if data.get('image') is not None:
                data['image'] = data['image'].resize((new_upscaled_width, new_upscaled_height))
                
            # Resize the inpainting mask if it exists
            if data.get('inpaintingMask') is not None:
                data['inpaintingMask'] = data['inpaintingMask'].resize((new_upscaled_width, new_upscaled_height))
                
            # Update the dimensions in validated_data to match the upscaled image
            validated_data['width'] = new_upscaled_width
            validated_data['height'] = new_upscaled_height
            
            # Update the image references in validated_data to point to the resized images
            validated_data['image'] = data['image']
            validated_data['image_data'] = data['image']
            validated_data['mask_data'] = data.get('inpaintingMask')
        

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
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_0.index(item) + 1, "queue_length": len(request_queue_0)}), 202

            elif data['gpu_id'] == 1:
                for item in request_queue_1:
                    if str(item.data['accountId']) == str(account_id):
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_1.index(item) + 1, "queue_length": len(request_queue_1)}), 202
                        
                        
                        
                        
                        
                        
        if data['ip'] is not None:
            if data['ip'] != "123123123":
                for item in request_queue_0:
                    if item.data['ip'] == data['ip']:
                        return jsonify({"status": item.status, "request_id": item.request_id, "position": request_queue_0.index(item) + 1, "queue_length": len(request_queue_0)}), 202
                        
                for item in request_queue_1:
                    if item.data['ip'] == data['ip']:
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
        if data['request_type'] != "img2video" and model_name not in diffusion_models:
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
        
        
        
        if data['request_type'] != "img2video":
            ckpt_name = diffusion_models[data['model']]['model_path']
            
            # remove models/ from the ckpt_name:
            ckpt_name = ckpt_name.replace("models/", "")
        else:
            # For img2video, we don't use traditional checkpoints
            ckpt_name = None
        
        # load it from the json:
        workflow_file = None
        if data['request_type'] == "txt2img":
            if data['model'].startswith("sd3"):
                workflow_file = 'SD3-txt2imgv2-1.json'
                with open ('SD3-txt2imgv2-1.json', 'r') as f:
                    # make it a dict:
                    request_json = json.load(f)
            else:
                workflow_file = 'txt2imgv3.json'
                with open('txt2imgv3.json', 'r') as f:
                    # make it a dict:
                    request_json = json.load(f)
        elif data['request_type'] == "img2img":
            workflow_file = 'img2imgv3.json'
            with open('img2imgv3.json', 'r') as f:
                # make it a dict:
                request_json = json.load(f)
        elif data['request_type'] == "img2video":
            workflow_file = 'img2video.json'
            with open('img2video.json', 'r') as f:
                # make it a dict:
                request_json = json.load(f)
        elif data['request_type'] == "inpainting":
            workflow_file = 'inpaintingv3.json'
            with open('inpaintingv3.json', 'r') as f:
                # make it a dict:
                request_json = json.load(f)
        
        print(f"DEBUG: Loaded workflow {workflow_file} for request_type: {data['request_type']}, model: {data.get('model', 'unknown')}")
        print(f"DEBUG: Workflow contains nodes: {list(request_json.keys())}")
                
        # print("Request JSON 288: ", request_json['288'])
        if data['request_type'] != "img2video" and ckpt_name is not None:
            request_json['288']['inputs']['ckpt_name'] = ckpt_name
                
        if data['request_type'] == "img2img" or data['request_type'] == "inpainting" or data['request_type'] == "img2video":
            request_json['301']['inputs']['value'] = data['strength']
            # convert the PIL Image to a base64 string:
            # print type:
            # make sure the directory exists:
            os.makedirs("process_images/image", exist_ok=True)
            request_json['307']['inputs']['image'] = f"C:\\Users\\anime\\Documents\\Coding\\JSCammie-Project\\JSCammie-SD-API\\process_images\\image\\image{request_id}{data['gpu_id']}.png"
        else:
            # txt2img, so set the strength to 1:
            request_json['301']['inputs']['value'] = 1
            
        if data['request_type'] == "inpainting":
            # make sure the dir exists:
            os.makedirs("process_images/inpainting", exist_ok=True)
            request_json['309']['inputs']['image'] = f"C:\\Users\\anime\\Documents\\Coding\\JSCammie-Project\\JSCammie-SD-API\\process_images\\inpainting\\inpainting{request_id}{data['gpu_id']}.png"
        
        if data['lora_strengths'] is None or data['lora_strengths'] == []:
            data['lora_strengths'] = []
            
        data['loras_broken'] = False
        
        load_loras(request_id, data['lora'], data)
        
        for key in data['loras_data']:
            print(key, data['loras_data'][key])
            
            # for each lora in the data['loras_data'] dict, add it to the request_json:
            for i, key in enumerate(data['loras_data']):
                # For img2video, apply LoRAs at 70% strength and to both model nodes
                if data['request_type'] == "img2video":
                    # Calculate 70% of the original strength, but keep lightning LoRA at full strength
                    original_strength = data['loras_data'][key][1]
                    
                    # Lightning LoRA should NOT be reduced - keep at full strength
                    if key == 'style-style-wanlightning':
                        adjusted_strength = original_strength
                        print(f"DEBUG: Lightning LoRA maintains full strength - {key}: {original_strength} (no reduction)")
                    else:
                        adjusted_strength = original_strength * 0.7
                        print(f"DEBUG: Adjusting LoRA strength for img2video - {key}: {original_strength} -> {adjusted_strength}")
                    
                    # Apply to both high noise model (node 117) and low noise model (node 326)
                    nodes_updated = []
                    if '117' in request_json:
                        request_json['117']['inputs'][f"lora_{i+1}"] = {
                            "on": True,
                            "lora": data['loras_data'][key][0],
                            "strength": adjusted_strength
                        }
                        nodes_updated.append("117 (high noise)")
                    
                    if '326' in request_json:
                        request_json['326']['inputs'][f"lora_{i+1}"] = {
                            "on": True,
                            "lora": data['loras_data'][key][0],
                            "strength": adjusted_strength
                        }
                        nodes_updated.append("326 (low noise)")
                    
                    print(f"DEBUG: Applied LoRA {key} to img2video nodes: {', '.join(nodes_updated)}")
                else:
                    # Standard LoRA application for other generation types
                    if '117' in request_json:
                        request_json['117']['inputs'][f"lora_{i+1}"] = {
                            "on": True,
                            "lora": data['loras_data'][key][0],
                            "strength": data['loras_data'][key][1]
                        }
                        print(f"DEBUG: Applied LoRA {key} to standard node 117")
                        
        # Prompt
        # Positive Prompt
        if '298' in request_json:
            request_json['298']['inputs']['text'] = data['prompt']
        else:
            print(f"WARNING: Node 298 (positive prompt) not found in workflow for request_type: {data['request_type']}")
        
        # Negative Prompt
        if '300' in request_json:
            request_json['300']['inputs']['text'] = data['negative_prompt']
        else:
            print(f"WARNING: Node 300 (negative prompt) not found in workflow for request_type: {data['request_type']}")
        
        # Steps - with safety checks
        if '303' in request_json:
            request_json['303']['inputs']['value'] = data['steps']
        else:
            print(f"WARNING: Node 303 (steps) not found in workflow for request_type: {data['request_type']}")
            
        if '305' in request_json:
            request_json['305']['inputs']['value'] = data['guidance']
        else:
            print(f"WARNING: Node 305 (guidance) not found in workflow for request_type: {data['request_type']}")
            
        if '304' in request_json:
            request_json['304']['inputs']['value'] = data['seed']
        else:
            print(f"WARNING: Node 304 (seed) not found in workflow for request_type: {data['request_type']}")
            
        # Dynamic batch_size for 2x images support (doubleImages extra) in img2video
        if data['request_type'] == "img2video" and '318' in request_json:
            # Dynamic batch_size for 2x images support (doubleImages extra)
            if data.get('extras', {}).get('doubleImages', False):
                batch_size = 2
                print(f"DEBUG: img2video doubleImages enabled - setting batch_size to {batch_size}")
            else:
                batch_size = 1
                print(f"DEBUG: img2video doubleImages disabled - using batch_size {batch_size}")
                
            request_json['318']['inputs']['batch_size'] = batch_size
            
            # Dynamic length setting from global settings
            video_length = global_settings.get('img2video_length', 81)
            request_json['318']['inputs']['length'] = video_length
            print(f"DEBUG: Set video length to {video_length} frames for img2video (from global settings)")
        elif data['request_type'] == "img2video":
            print(f"WARNING: Node 318 (WanImageToVideo) not found in img2video workflow")
            
        # Dynamic upscaling for img2video (node 316 - Image Resize by Longer Side)
        if data['request_type'] == "img2video" and '316' in request_json:
            base_size = 480  # Default base size
            
            # Check if upscaling is enabled in extras
            if data.get('extras', {}).get('upscale', False):
                upscale_ratio = global_settings.get('img2video_upscale_ratio', 1.5)
                upscaled_size = calculate_img2video_upscale_size(base_size, upscale_ratio)
                print(f"DEBUG: img2video upscaling enabled - base_size: {base_size}, ratio: {upscale_ratio}, upscaled_size: {upscaled_size}")
            else:
                upscaled_size = base_size
                print(f"DEBUG: img2video upscaling disabled - using base_size: {base_size}")
                
            request_json['316']['inputs']['size'] = upscaled_size
        elif data['request_type'] == "img2video":
            print(f"WARNING: Node 316 (Image Resize) not found in img2video workflow")
        
        # Only modify node 285 for non-img2video requests (img2video uses different workflow nodes)
        if data['request_type'] != "img2video":
            if '285' in request_json:
                request_json['285']['inputs']['width'] = data['width']
                request_json['285']['inputs']['height'] = data['height']
            else:
                print(f"ERROR: Node 285 not found in workflow for request_type: {data['request_type']}, model: {data.get('model', 'unknown')}")
                print(f"Available nodes: {list(request_json.keys())}")
                # Try to find an alternative node for width/height
                for node_id, node_data in request_json.items():
                    if node_id.isnumeric() and 'inputs' in node_data:
                        if 'width' in node_data['inputs'] and 'height' in node_data['inputs']:
                            print(f"Found alternative node {node_id} with width/height inputs")
                            request_json[node_id]['inputs']['width'] = data['width']
                            request_json[node_id]['inputs']['height'] = data['height']
                            break
            
        if '306' in request_json:
            request_json['306']['inputs']['value'] = data['image_count']
        else:
            print(f"WARNING: Node 306 (image_count) not found in workflow for request_type: {data['request_type']}")
            
        # Only modify node 64 for non-img2video requests (img2video uses different workflow nodes)
        if data['request_type'] != "img2video":
            if '64' in request_json:
                request_json['64']['inputs']['filename_prefix'] = request_id
            else:
                print(f"WARNING: Node 64 (filename_prefix) not found in workflow for request_type: {data['request_type']}")
            
        if not data['model'].startswith("pdxl") and not data['model'].startswith("flux") and not data['model'].startswith("illustrious"):
            request_json['300']['inputs']['text'] = f"{request_json['300']['inputs']['text']} embedding:boring_e621_v4.pt embedding:fluffynegative.pt embedding:badyiffymix41.safetensors embedding:gnarlysick-neg.pt embedding:negative_hand-neg.pt"
                    
        # print(f"LIGHNING MODE: {data['lightning_mode']}")
                    
        # Scheduler and Sampler settings - handle different node IDs for different generation types
        if data['request_type'] == "img2video":
            # For img2video, use global settings for sampler and scheduler
            sampler_nodes = ['322', '323']
            img2video_sampler = global_settings.get('img2video_sampler', 'lcm')
            img2video_scheduler = global_settings.get('img2video_scheduler', 'simple')
            
            for node_id in sampler_nodes:
                if node_id in request_json:
                    request_json[node_id]['inputs']['sampler_name'] = img2video_sampler
                    request_json[node_id]['inputs']['scheduler'] = img2video_scheduler
                    print(f"DEBUG: Applied img2video sampler settings to node {node_id}: sampler={img2video_sampler}, scheduler={img2video_scheduler}")
                else:
                    print(f"WARNING: KSampler node {node_id} not found in img2video workflow")
        elif data['model'].startswith("flux"):
            # Flux model specific settings
            sampler_nodes = ['282']
            for node_id in sampler_nodes:
                if node_id in request_json:
                    request_json[node_id]['inputs']['sampler_name'] = "euler"
                    request_json[node_id]['inputs']['scheduler'] = "normal"
        else:
            # Determine which sampler nodes to use based on generation type
            sampler_nodes = ['282']
            
            # Apply scheduler settings to all relevant sampler nodes
            for node_id in sampler_nodes:
                if node_id in request_json:
                    if data['scheduler'] == "lcm":
                        request_json[node_id]['inputs']['sampler_name'] = "lcm"
                        request_json[node_id]['inputs']['scheduler'] = "normal"
                    elif data['scheduler'] == "ddim_video":
                        request_json[node_id]['inputs']['sampler_name'] = "ddim"
                        request_json[node_id]['inputs']['scheduler'] = "beta"
                    elif data['scheduler'] == "eulera":
                        request_json[node_id]['inputs']['sampler_name'] = "euler_ancestral"
                        request_json[node_id]['inputs']['scheduler'] = global_settings['ksampler_scheduler']
                    elif data['scheduler'] == "dpm":
                        request_json[node_id]['inputs']['sampler_name'] = "dpmpp_2m"
                        request_json[node_id]['inputs']['scheduler'] = global_settings['ksampler_scheduler']
                    elif data['scheduler'] == "ddim":
                        request_json[node_id]['inputs']['sampler_name'] = "ddim"
                        request_json[node_id]['inputs']['scheduler'] = global_settings['ksampler_scheduler']
                    elif data['scheduler'] == "normal":
                        # For normal scheduler, keep default sampler settings
                        request_json[node_id]['inputs']['scheduler'] = "normal"
                else:
                    print(f"WARNING: KSampler node {node_id} not found in workflow for request_type: {data['request_type']}")

                
                
                
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
                "image": f"C:\\Users\\anime\\Documents\\Coding\\JSCammie-Project\\JSCammie-SD-API\\process_images\\image\\image{request_id}{data['gpu_id']}.png",
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
                        
        if extras['removeBackground'] == True and data['request_type'] != "img2video":
            
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
        
        
        # inject the following into the request_json, making sure that anything that uses 288 model, uses this instead:
        # "14": {
        #     "inputs": {
        #     "model": [
        #         "1",
        #         0
        #     ]
        #     },
        #     "class_type": "Mahiro",
        #     "_meta": {
        #     "title": "Mahiro is so cute that she deserves a better guidance function!! (。・ω・。)"
        #     }
        # }
              
        # Find the highest node ID to create a new one for Mahiro
        nextAvailableId = 0
        for key in request_json:
            if key.isnumeric():
                if int(key) > nextAvailableId:
                    nextAvailableId = int(key)
                    
        nextAvailableId += 1  # Increment to get the next available ID
        mahiroId = str(nextAvailableId)
        
        # Create the Mahiro node with direct connection to the checkpoint loader
        if data['request_type'] == "img2video":
            model_id = "312"  # ModelSamplingSD3 output, not UnetLoaderGGUF directly
        else:
            model_id = "288"
            
        request_json[mahiroId] = {
            "inputs": {
                "model": [
                    f"{model_id}",  # Checkpoint loader
                    0       # First output (model)
                ]
            },
            "class_type": "Mahiro",
            "_meta": {
                "title": "Mahiro is so cute that she deserves a better guidance function!! (。・ω・。)"
            }
        }
        
        # Find all nodes that directly use the model output from the checkpoint loader
        # and redirect them to use Mahiro's output instead
        for key in request_json:
            if key.isnumeric() and key != mahiroId:  # Skip the Mahiro node itself
                if "inputs" in request_json[key] and "model" in request_json[key]["inputs"]:
                    input_val = request_json[key]["inputs"]["model"]
                    if isinstance(input_val, list) and len(input_val) >= 2:
                        if data['request_type'] == "img2video":
                            if input_val[0] == "312" and input_val[1] == 0:
                                request_json[key]["inputs"]["model"] = [mahiroId, 0]
                        elif data['request_type'] != "img2video":
                            if input_val[0] == "288" and input_val[1] == 0:  # Only redirect model connections
                                # Update to use Mahiro's output
                                request_json[key]["inputs"]["model"] = [mahiroId, 0]
                            
        # Check if upscaling should be disabled when LazyCache is enabled
        disable_upscale_for_easycache = (
            global_settings.get('easycache_enabled', True) and 
            global_settings.get('easycache_disable_upscaling', True) and
            (data['request_type'] != "img2video" or global_settings.get('easycache_img2video_enabled', False))
        )
        
        if disable_upscale_for_easycache and extras['upscale'] == True:
            print("DEBUG: Upscaling disabled because LazyCache is enabled and easycache_disable_upscaling is True")
        
        if extras['upscale'] == True and data['request_type'] != "img2video" and not disable_upscale_for_easycache:
            
            # detect highest node ID in the request_json:
            highest_node_id = 0
            for key in request_json:
                # check if its an int before comparing:
                if key.isnumeric():
                    if int(key) > highest_node_id:
                        highest_node_id = int(key)
            
            upscale_hidiffusion_node = {
                "inputs": {
                    "model_type": "auto",
                    "res_mode": "high (1536-2048)",
                    "upscale_mode": "bicubic",
                    "ca_upscale_mode": "bicubic",
                    "model": [
                        f"{mahiroId}",
                        0
                    ]
                },
                "class_type": "ApplyRAUNetSimple",
                "_meta": {
                    "title": "ApplyRAUNetSimple"
                }
            }
            
            # add the upscale_hidiffusion_node to the request_json:
            request_json[str(highest_node_id + 1)] = upscale_hidiffusion_node
            
            # change the width and height of 285 to the new width and height (only for txt2img):
            if '285' in request_json and 'inputs' in request_json['285']:
                request_json['285']['inputs']['width'] = data['width'] * data['upscale_scale']
                request_json['285']['inputs']['height'] = data['height'] * data['upscale_scale']
            
            # multiply the steps by the upscale_steps, ensure its an int with rounding:
            request_json['303']['inputs']['value'] = int(round(data['steps'] * data['upscale_steps']))
            
            # Find all nodes that directly use the model output from mahiro
            # and redirect them to use the upscale_hidiffusion_node instead
            for key in request_json:
                if key.isnumeric() and key != mahiroId and key != str(highest_node_id + 1):
                    if "inputs" in request_json[key] and "model" in request_json[key]["inputs"]:
                        input_val = request_json[key]["inputs"]["model"]
                        if isinstance(input_val, list) and len(input_val) >= 2:
                            if input_val[0] == mahiroId and input_val[1] == 0:  # Only redirect model connections
                                # Update to use upscale_hidiffusion_node's output
                                request_json[key]["inputs"]["model"] = [str(highest_node_id + 1), 0]
        
        else:
            # apply the hidiffusion_node to the request_json WITHOUT changing the width and height of 285
            # Skip hidiffusion for img2video as it uses a different workflow
            if data['request_type'] != "img2video":
                
                # detect highest node ID in the request_json:
                highest_node_id = 0
                for key in request_json:
                    # check if its an int before comparing:
                    if key.isnumeric():
                        if int(key) > highest_node_id:
                            highest_node_id = int(key)
                            
                hidiffusion_node = {
                    "inputs": {
                        "model_type": "auto",
                        "res_mode": "low (1024 or lower)",
                        "upscale_mode": "bicubic",
                        "ca_upscale_mode": "bicubic",
                        "model": [
                            f"{mahiroId}",
                            0
                        ]
                    },
                    "class_type": "ApplyRAUNetSimple",
                    "_meta": {
                        "title": "ApplyRAUNetSimple"
                    }
                }
                
                # add the hidiffusion_node to the request_json:
                request_json[str(highest_node_id + 1)] = hidiffusion_node
                
                # Find all nodes that directly use the model output from mahiro
                # and redirect them to use the hidiffusion_node instead
                for key in request_json:
                    if key.isnumeric() and key != mahiroId and key != str(highest_node_id + 1):
                        if "inputs" in request_json[key] and "model" in request_json[key]["inputs"]:
                            input_val = request_json[key]["inputs"]["model"]
                            if isinstance(input_val, list) and len(input_val) >= 2:
                                if input_val[0] == mahiroId and input_val[1] == 0:  # Only redirect model connections
                                    # Update to use hidiffusion_node's output
                                    request_json[key]["inputs"]["model"] = [str(highest_node_id + 1), 0]

                            
        # "312": {
        #     "inputs": {
        #     "vae_name": "taesdxl"
        #     },
        #     "class_type": "VAELoader",
        #     "_meta": {
        #     "title": "Load VAE"
        #     }
        # },
        
        # # Find the highest node ID to create a new one for VAE Loader
        # nextAvailableId = 0
        # for key in request_json:
        #     if key.isnumeric():
        #         if int(key) > nextAvailableId:
        #             nextAvailableId = int(key)
                    
        # nextAvailableId += 1  # Increment to get the next available ID
        # vaeLoaderId = str(nextAvailableId)
        
        # vaeToLoad = "taesd"
        # # if the string model includes a -, then set it to taesdxl:
        # if "-" in data['model']:
        #     vaeToLoad = "taesdxl"

        # # Create the VAE Loader node with direct connection to the checkpoint loader
        # request_json[vaeLoaderId] = {
        #     "inputs": {
        #         "vae_name": vaeToLoad
        #     },
        #     "class_type": "VAELoader",
        #     "_meta": {
        #         "title": "Load VAE"
        #     }
        # }
        
        # # Find all nodes that directly use the vae output from the checkpoint loader
        # for key in request_json:
        #     if key.isnumeric() and key != vaeLoaderId:  # Skip the VAE Loader node itself
        #         if "inputs" in request_json[key] and "vae" in request_json[key]["inputs"]:
        #             input_val = request_json[key]["inputs"]["vae"]
        #             if isinstance(input_val, list) and len(input_val) >= 2:
        #                 if input_val[0] == "288" and input_val[1] == 2:  # VAE connections use port 2
        #                     request_json[key]["inputs"]["vae"] = [vaeLoaderId, 0]
        
        # For img2video, ensure node 288 is removed if it somehow got added
        if data['request_type'] == "img2video" and '288' in request_json:
            print("WARNING: Found node 288 in img2video workflow, removing it")
            del request_json['288']
        
        # Add LazyCache node if enabled
        if global_settings.get('easycache_enabled', True):
            # Check if LazyCache should be enabled for img2video (default off)
            if data['request_type'] == "img2video" and not global_settings.get('easycache_img2video_enabled', False):
                pass  # Skip LazyCache for img2video when disabled
            else:
                # Find the highest node ID to create a new one for LazyCache
                easycache_node_id = 0
                for key in request_json:
                    if key.isnumeric():
                        if int(key) > easycache_node_id:
                            easycache_node_id = int(key)
                
                easycache_node_id += 1  # Increment to get the next available ID
                easycache_id = str(easycache_node_id)
                
                # Determine which model node to connect to
                # For compatibility with Mahiro, check if Mahiro node exists and connect to it
                # Otherwise connect to the base model nodes
                if data['request_type'] == "img2video":
                    # Look for Mahiro node first, then fallback to base node
                    mahiro_node = None
                    for key in request_json:
                        if (key.isnumeric() and 
                            request_json[key].get('class_type') == 'Mahiro'):
                            mahiro_node = key
                            break
                    
                    model_source_id = mahiro_node if mahiro_node else "311"  # UnetLoaderGGUF for img2video
                else:
                    # Look for Mahiro node first, then fallback to base node  
                    mahiro_node = None
                    for key in request_json:
                        if (key.isnumeric() and 
                            request_json[key].get('class_type') == 'Mahiro'):
                            mahiro_node = key
                            break
                    
                    model_source_id = mahiro_node if mahiro_node else "288"  # Checkpoint loader for regular generations
                
                # Create the LazyCache node
                request_json[easycache_id] = {
                    "inputs": {
                        "reuse_threshold": global_settings.get('easycache_reuse_threshold', 0.2),
                        "start_percent": global_settings.get('easycache_start_percent', 0.15),
                        "end_percent": global_settings.get('easycache_end_percent', 0.95),
                        "verbose": global_settings.get('easycache_verbose', False),
                        "model": [
                            model_source_id,
                            0
                        ]
                    },
                    "class_type": "LazyCache",
                    "_meta": {
                        "title": "LazyCache"
                    }
                }
                
                # Find all nodes that use the model output from the source node
                # and redirect them to use LazyCache's output instead
                redirected_nodes = []
                for key in request_json:
                    if (key.isnumeric() and 
                        key != easycache_id and  # Skip the LazyCache node itself
                        request_json[key].get('class_type') != 'LazyCache'):  # Prevent circular references
                        if "inputs" in request_json[key] and "model" in request_json[key]["inputs"]:
                            input_val = request_json[key]["inputs"]["model"]
                            if isinstance(input_val, list) and len(input_val) >= 2:
                                if input_val[0] == model_source_id and input_val[1] == 0:
                                    # Validate this won't create a circular reference
                                    if key != model_source_id:  # Prevent self-reference
                                        # Update to use LazyCache's output
                                        request_json[key]["inputs"]["model"] = [easycache_id, 0]
                                        redirected_nodes.append(key)
                                        print(f"DEBUG: Redirected model connection in node {key} to LazyCache node {easycache_id}")
                                    else:
                                        print(f"WARNING: Prevented circular reference - node {key} tried to reference itself")
                
                print(f"DEBUG: Added LazyCache node {easycache_id} connecting to {model_source_id}, redirected {len(redirected_nodes)} nodes")
        
        data['request_json'] = request_json
        
        # Debug: Print the request_json for img2video to see what's being sent
        if data['request_type'] == "img2video":
            print("DEBUG: img2video request_json nodes:")
            for node_id in sorted(request_json.keys()):
                if node_id.isnumeric():
                    node = request_json[node_id]
                    print(f"  Node {node_id}: {node.get('class_type', 'Unknown')}")
                    if 'inputs' in node:
                        for input_name, input_value in node['inputs'].items():
                            if isinstance(input_value, list) and len(input_value) >= 2:
                                print(f"    {input_name}: references node {input_value[0]}")
        
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
            data['images_dir'] = "C:/Users/anime/Documents/Coding/ComfyUIs/ComfyUI-P/ComfyUI/output"
            position = len(request_queue_0)  # Current position in the queue is its length
            
        elif data['gpu_id'] == 1:
            queueNumber = 1
            data['server_address'] = "http://127.0.0.1:8188"
            data['images_dir'] = "C:/Users/anime/Documents/Coding/ComfyUIs/ComfyUI-P/ComfyUI/output"
            # data['server_address'] = "http://127.0.0.1:8189"
            # data['images_dir'] = "C:/Users/anime/Documents/Coding/Stability Matrix/Data/Packages/ComfyUI2/output"
            position = len(request_queue_1)  # Current position in the queue is its length
            
                
        if data.get('fastqueue', False) is True:
            fastqueue = True
        
        positionToSet = 0
        
        if fastqueue == True:
            if queueNumber == 0:
                # If account ID is "1", put at the very front
                if str(data['accountId']) == "1":
                    request_queue_0.insert(0, queue_item)
                    positionToSet = 0
                else:
                    # if someone is already using the fastqueue, then make their position above the fastqueue'er(s):
                    # but skip past any account ID "1" items first
                    for item in request_queue_0:
                        if str(item.data.get('accountId', '')) == "1":
                            positionToSet += 1
                        elif item.data.get('fastqueue', False) == True:
                            positionToSet += 1
                        else:
                            break

                    request_queue_0.insert(positionToSet, queue_item)
            
            elif queueNumber == 1:
                # If account ID is "1", put at the very front
                if str(data['accountId']) == "1":
                    request_queue_1.insert(0, queue_item)
                    positionToSet = 0
                else:
                    # if someone is already using the fastqueue, then make their position above the fastqueue'er(s):
                    # but skip past any account ID "1" items first
                    for item in request_queue_1:
                        if str(item.data.get('accountId', '')) == "1":
                            positionToSet += 1
                        elif item.data.get('fastqueue', False) == True:
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
    "ip": "123123123",
    "fastqueue": False
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