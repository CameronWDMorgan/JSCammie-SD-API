import diffusers
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline
import numpy as np
import torch
import PIL
from PIL import Image
import time
import cv2
import os
import DeepCache
from DeepCache import DeepCacheSDHelper
import transformers



print(f"Number of CUDA devices: {torch.cuda.device_count()}")

def load_pipeline():
    
    controlnet = diffusers.ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
    
    # pipeline = StableDiffusionImg2ImgPipeline.from_single_file('models/fluffysonic_v20.safetensors', torch_dtype=torch.float16, variant="fp16")
    pipeline = diffusers.StableDiffusionControlNetPipeline.from_single_file(
        'models/fluffysonic_v20.safetensors',
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        num_in_channels=4,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
        )

    # Ensure sampler uses "trailing" timesteps.
    # pipeline.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    
    pipeline.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()

    pipeline.to('cuda:1')
    
    helper = DeepCache.DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(
        cache_interval=2,
        cache_branch_id=0,
    )
    helper.enable()

    start_time = time.time()

    # load lora:
    # pipeline.load_lora_weights(
    #     'loras/style/bitdiffuserx.safetensors', 
    #     low_cpu_mem_usage=False,
    #     ignore_mismatched_sizes=True
    # )

    time_to_load_lora = time.time() - start_time
    print(f"Time to load LORA: {time_to_load_lora}")
        
    return pipeline

# Load the video
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

# Create a directory to store the frames
frame_dir = 'video_frames'
os.makedirs(frame_dir, exist_ok=True)

max_frames_wanted = 12

original_fps = cap.get(cv2.CAP_PROP_FPS)
orignal_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Extract frames
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if frame_count == max_frames_wanted:
#         break
#     if not ret:
#         break  # Break the loop if there are no frames left to read

#     # Save each frame as a PNG file
#     frame_filename = os.path.join(frame_dir, f'frame_{frame_count:04d}.png')
#     cv2.imwrite(frame_filename, frame)
#     frame_count += 1

# rewrite the above code, but let me skip x frames after any extracted frames:
frame_count = 0
extracted_frames = 0
max_frames_wanted = 8
skip_frames = 1  # Number of frames to skip after extracting one

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop if there are no frames left to read

    # Check if we have reached the maximum number of frames we want to extract
    if extracted_frames >= max_frames_wanted:
        break

    # Only save every (skip_frames + 1)th frame
    if frame_count % (skip_frames + 1) == 0:
        # Save the current frame as a PNG file
        frame_filename = os.path.join(frame_dir, f'frame_{extracted_frames:04d}.png')
        cv2.imwrite(frame_filename, frame)
        extracted_frames += 1

    frame_count += 1

cap.release()  # Release the video capture object

print(f'Frames are saved in {frame_dir}, total {extracted_frames} frames extracted.')



# make the video have the shortest side be 512 pixels, the other side will be > 512 scaled to keep the aspect ratio:
# make sure they are divisible by 8:
shortest_side = min(orignal_size)
scale_factor = 512 / shortest_side
width_height = (int(orignal_size[0] * scale_factor), int(orignal_size[1] * scale_factor))
# make sure they are divisible by 8:
width_height = (width_height[0] - width_height[0] % 8, width_height[1] - width_height[1] % 8)

# Assuming `extracted_frames` is the total number of frames actually saved
for i in range(extracted_frames):
    frame = cv2.imread(f'video_frames/frame_{i:04d}.png')
    if frame is not None:
        resized_frame = cv2.resize(frame, width_height, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'video_frames/frame_{i:04d}.png', resized_frame)
    else:
        print(f"Warning: No frame found at index {i}, expected due to skipping.")

    
# Load the frames
frames = [Image.open(f'video_frames/frame_{i:04d}.png') for i in range(extracted_frames)]

prompt = "((masterpiece, high quality, highres, emphasis lines)), 1girl, solo, (mobian, amy rose, green eyes), nude, facing viewer, looking at viewer, big breasts, thick thighs"
negative_prompt = "monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective"

seed = 123123123

img2img_outputs = []

pipeline = load_pipeline()

os.makedirs('outputs/vid2vid', exist_ok=True)

low_threshold = 100
high_threshold = 200
    
for i in range(extracted_frames):
    frame = frames[i]
    frame = frame.convert('RGB')
    original_image = frame.resize(width_height, Image.LANCZOS)
    # debug printing:
    print(f"Processing frame {i} with size {frame.size}\n {frame.getpixel((0, 0))} {frame.getpixel((frame.size[0] - 1, frame.size[1] - 1))}")
    
    image = np.array(original_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # check the size of the canny image:
    print(f"Canny image size: {canny_image.size}")\
        
    # ensure the canny image is PIL image:
    canny_image = canny_image.convert('RGB')
    canny_image = canny_image.resize(width_height, Image.LANCZOS)

    generator = torch.Generator(device='cuda:1')
    generator.manual_seed(seed)
    
    # Ensure single image and prompt are passed to the pipeline:
    output = pipeline(
        prompt=prompt,  # Make sure this is designed for a single frame
        negative_prompt=negative_prompt,
        width=width_height[0],
        height=width_height[1],
        eta=1.0,
        image=canny_image,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    output.save(f'outputs/vid2vid/frame_{i:04d}.png')

# num_images = min(4, extracted_frames - i)
# images = pipeline(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     width=width_height[0],
#     height=width_height[1],
#     image=[frame] * num_images,
#     num_images_per_prompt=num_images,
#     num_inference_steps=20,
#     strength=0.6,
#     guidance_scale=7.5,
# ).images[0]

# # generate 1-4 images at once depending on how many frames are left, looping it until all frames are processed:

# for i in range(0, extracted_frames, 4):
#     num_images = min(4, extracted_frames - i)
#     images = pipeline(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         width=width_height[0],
#         height=width_height[1],
#         image=[frames[j] for j in range(i, i + num_images)],
#         num_images_per_prompt=num_images,
#         num_inference_steps=40,
#         strength=0.2,
#         guidance_scale=7.5,
#     ).images

#     for j in range(num_images):
#         # images is an array of pil images, save each one:
#         images[j].save(f'outputs/vid2vid/frame_{i + j:04d}.png')

    # print(f"Processed frames {i} to {i + num_images - 1}")
    
print("All frames processed.")



    
    
    
# convert the frames to a video:

frame_array = []
for i in range(extracted_frames):
    img = cv2.imread(f'outputs/vid2vid/frame_{i:04d}.png')
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter(f'outputs/vid2vid/output-{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), original_fps, size, True)

for frame in frame_array:
    out.write(frame)

out.release()

# image = pipeline(
#     prompt="((masterpiece, high quality, highres, emphasis lines)), 1girl, amy rose, denim jeans, white crop top, in city",
#     negative_prompt="nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
#     num_inference_steps=20,
#     guidance_scale=7.5,
#     num_images_per_prompt=1,
# ).images[0]

# os.makedirs('outputs/test', exist_ok=True)

# Save the image, making sure it has unique filename:
# image.save(f'outputs/test/{time.time()}.png')