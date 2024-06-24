import base64
import io
import diffusers
import flask
from flask import request, jsonify
from flask_cors import CORS
import torch
import PIL
from PIL import Image
import uuid
import time
import os


queue = []
results = {}

queue_busy = False

torch_dtype_value = torch.float16

class QueueRequest:
    def __init__(self, request_id, data, image_data=None):
        self.request_id = request_id
        self.data = data
        self.status = "queued"

app = flask.Flask(__name__)
CORS(app)

pipelines = {
    "realisticvision": {'path': "models/realisticVisionV60B1_v51VAE.safetensors", 'loaded': None},
}

def process_request(queue_item):
    data = queue_item.data
    
    base64_image_array = []
    
    # if the model isnt loaded, load it:
    if pipelines[data['model']]['loaded'] is None:
        pipeline = diffusers.StableDiffusionPipeline.from_single_file(
            pipelines[data['model']]['path'],
            torch_dtype=torch_dtype_value
        )
        pipeline.enable_vae_slicing()
        pipeline.enable_vae_tiling()
        
        pipelines[data['model']]['loaded'] = pipeline

    pipeline = pipelines[data['model']]['loaded']
    
    # set the scheduler:
    pipeline.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    
    pipeline.to('cuda:1')
    
    image = pipeline(
        prompt=data['prompt'],
        negative_prompt=data['negative_prompt'],
        width=data['width'],
        height=data['height'],
        num_inference_steps=20,
        num_images_per_prompt=1,
    ).images[0]
    
    pipeline.to('cpu')
    
    os.makedirs('outputs', exist_ok=True)
    
    image.save(f'outputs/output-{queue_item.request_id}.png')
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    base64_image_array.append(img_str)
    
    # append the result to the results list, with the request_id as the key:
    results[queue_item.request_id] = {
        "images": base64_image_array,
    }
        
    queue_item.status = "completed"
    return "processed"

def process_queue():
    global queue_busy
    
    while True:
        time.sleep(0.01)  # Short sleep to prevent CPU overutilization
        if not queue_busy:
                        
            if queue:  # Check if the queue is not empty
                queue_item = queue[0]  # Get the first item
                if queue_item.status == "queued":
                    queue_item.status = "waiting"
                    queue_busy = True
                    
                    result = process_request(queue_item)
                    
                    queue_busy = False
                    
                    if result == "processed":
                        queue.remove(queue_item)  # Remove the item from the queue
                    elif queue_item.status in ["completed", "error", "skipped"]:
                        queue.remove(queue_item)  # Remove the item from the queue
            
        # Sleep if no unprocessed request is found
        if not any(item.status == "queued" for item in queue):
            time.sleep(0.5)

# generate endpoint:
@app.route('/generate', methods=['POST'])
def generate():
    
    data = request.json
    print(data)
    
    # sanitize the data:
    data['width'] = int(data['width'])
    data['height'] = int(data['height'])
    
    request_id = str(uuid.uuid4())
    queue_item = QueueRequest(request_id, data)
    
    queue.append(queue_item)
    
    position = len(queue)
    
    return jsonify({"status": "queued", "request_id": request_id, "position": position, "queue_length": position}), 202

@app.route('/queue_position/<request_id>', methods=['GET'])
def check_queue_position(request_id):

    for index, item in enumerate(queue):
        if item.request_id == request_id:
            return jsonify({"status": "waiting", "request_id": request_id, "position": index + 1, "queue_length": len(queue)}), 200
        
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

# test the endpoint:
def test_generate(data):
    response = app.test_client().post('/generate', json={
        "model": data['model'],
        "prompt": data['prompt'],
        "negative_prompt": data['negative_prompt'],
        "width": data['width'],
        "height": data['height'],
    })
    
test_generate({
    "model": "realisticvision",
    "prompt": "((masterpiece, high quality, highres, emphasis lines)), 1girl, black hair, yellow eyes, white dress, in city",
    "negative_prompt": "nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "width": 512,
    "height": 512,
})

test_generate({
    "model": "realisticvision",
    "prompt": "((masterpiece, high quality, highres, emphasis lines)), 1girl, blue hair, red eyes, black dress, in farm",
    "negative_prompt": "nsfw, monochrome, black and white, worst quality, low quality, watermark, signature, bad anatomy, bad hands, deformed limbs, blurry, cropped, cross-eyed, extra arms, extra legs, extra limbs, extra pupils, bad proportions, poorly drawn hands, simple background, bad background, bad lighting, bad perspective",
    "width": 512,
    "height": 512,
})

if __name__ == '__main__':
    import threading
    threading.Thread(target=process_queue, daemon=True).start()
    app.run(port=5678)