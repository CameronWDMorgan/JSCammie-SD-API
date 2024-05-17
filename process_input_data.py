from main import generate_error_response, blockedwords, round_to_multiple_of_eight, Image, random, io, base64, time
# randomize_string
def validate_input_data(data):
    validate_start_time = time.time()
    true_prompt = data['prompt']
    data['prompt'] = data['prompt'].replace('\r', '').replace('\n', '')
    
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

    data['accountId'] = data.get('accountId', 0)

    if data['accountId'] == "":
        data['accountId'] = 0

    data['accountId'] = int(data['accountId'])

    # data['prompt'] = randomize_string(data['prompt'])

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
        # if int(data['steps']) > 50:
        #     return None, "text 2 video is limited to 50 steps! Please reduce the number of steps and try again."


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
    
    # remove colons from prompt and negative prompt strings:
    data['prompt'] = data['prompt'].replace(":", "")
    data['negative_prompt'] = data['negative_prompt'].replace(":", "")
        
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
            
            # Determine the scaling factor to ensure both sides are at least 512px
            scale_factor = max(512 / data['image'].width, 512 / data['image'].height)
            
            # print("Image width height after")

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
        
    og_seed = data['seed']

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
        'upscale': data.get('upscale', False),
        'inpainting_original_option': True,
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
        'og_seed': int(og_seed),
        "save_image": bool(data.get("save_image", False)),
        "gpu_id": int(data.get("gpu_id", 0)),
    }
    
    if validated_data['request_type'] == "inpainting":
        validated_data['strength'] = 0.75
        

    return validated_data, None