import diffusers
import accelerate
from accelerate import Accelerator
import hidiffusion
import torch
import tomesd

from main import model_move_manual

from hidiffusion import apply_hidiffusion

from DeepCache import DeepCacheSDHelper

print(f"Number of CUDA devices: {torch.cuda.device_count()}")

def load_embeddings(pipeline, name):
    if not name.startswith("sdxl-"):
        pipeline.load_textual_inversion("embeddings/boring_e621_v4.pt")
        pipeline.load_textual_inversion("embeddings/fluffynegative.pt")
        pipeline.load_textual_inversion("embeddings/badyiffymix41.safetensors")
        pipeline.load_textual_inversion("embeddings/gnarlysick-neg.pt")
        pipeline.load_textual_inversion("embeddings/negative_hand-neg.pt")
    # else:
        # pipeline.load_textual_inversion("embeddings/sdxl/zPDXL.pt")
        # pipeline.load_textual_inversion("embeddings/sdxl/zPDXL-neg.pt")
    return pipeline

def enable_deep_cache(pipeline):
    pipeline.helper = DeepCacheSDHelper(pipe=pipeline)
    pipeline.helper.set_params(
        cache_interval=2,
        cache_branch_id=0,
    )
    pipeline.helper.enable()

torch_dtype = torch.float16

def txt2img(name, data, model_path):
    
    data['prev_same_lora'] = False
    data['prev_same_strengths'] = False
    data['prev_same_model'] = False

    if name.startswith("sdxl-"):
        pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = diffusers.StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    
    # if name.startswith("sdxl-"):
    #     pipeline = diffusers.DiffusionPipeline.from_pretrained(
    #         'models/saved/' + name,
    #         torch_dtype=torch_dtype
    #     )
    # else:
    #     pipeline = diffusers.DiffusionPipeline.from_pretrained(
    #         'models/saved/' + name,
    #         torch_dtype=torch_dtype,
    #         safety_checker=None
    #     )

    if model_move_manual:
        if data['gpu_id'] == 1:
            pipeline.to('cuda:1')
        if data['gpu_id'] == 0:
            pipeline.to('cuda:0')
    else:
        if data['gpu_id'] == 1:
            pipeline.enable_model_cpu_offload(gpu_id=1)
        if data['gpu_id'] == 0:
            pipeline.enable_model_cpu_offload(gpu_id=0)

    enable_deep_cache(pipeline)
                
    pipeline = load_embeddings(pipeline, name)
    
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
        
    tomesd.apply_patch(pipeline, ratio=0.2)
        
    return pipeline





def img2img(name, data, model_path):
    
    data['prev_same_lora'] = False
    data['prev_same_strengths'] = False
    data['prev_same_model'] = False
    

        
    # if name.startswith("sdxl-"):
    #     if txt2img_models[name]['loaded'] is None:
    #         txt2img_models[name]['loaded'] = txt2img(name, data, model_path)
            
    #     components = txt2img_models[name]['loaded'].components
    #     pipeline = diffusers.StableDiffusionXLImg2ImgPipeline(**components)
    # else:
    #     # load the pipeline manually:
    #     pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(
    #         model_path,
    #         torch_dtype=torch_dtype,
    #         safety_checker=None
    #     )
    #     pipeline = load_embeddings(pipeline, name)

    if name.startswith("sdxl-"):
        pipeline = diffusers.StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

    if model_move_manual:
        if data['gpu_id'] == 1:
            pipeline.to('cuda:1')
        if data['gpu_id'] == 0:
            pipeline.to('cuda:0')
    else:
        if data['gpu_id'] == 1:
            pipeline.enable_model_cpu_offload(gpu_id=1)
        if data['gpu_id'] == 0:
            pipeline.enable_model_cpu_offload(gpu_id=0)

    enable_deep_cache(pipeline)
                
    pipeline = load_embeddings(pipeline, name)
    
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
        
    tomesd.apply_patch(pipeline, ratio=0.2)
        
    return pipeline




def inpainting(name, data, model_path):
    
    data['prev_same_lora'] = False
    data['prev_same_strengths'] = False
    data['prev_same_model'] = False
    
    if name.startswith("sdxl-"):
        pipeline = diffusers.StableDiffusionXLInpaintPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            num_in_channels=4,
        )
    else:
        pipeline = diffusers.StableDiffusionInpaintPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            num_in_channels=4,
        )

    if model_move_manual:
        if data['gpu_id'] == 1:
            pipeline.to('cuda:1')
        if data['gpu_id'] == 0:
            pipeline.to('cuda:0')
    else:
        if data['gpu_id'] == 1:
            pipeline.enable_model_cpu_offload(gpu_id=1)
        if data['gpu_id'] == 0:
            pipeline.enable_model_cpu_offload(gpu_id=0)

    enable_deep_cache(pipeline)
                
    pipeline = load_embeddings(pipeline, name)
    
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
        
    tomesd.apply_patch(pipeline, ratio=0.2)
        
    return pipeline



















# adapter = diffusers.MotionAdapter.from_pretrained(
#     "guoyww/animatediff/mm_sd_v15_v2",
#     low_cpu_mem_usage=False,
#     device_map=None,
#     )
# sdxladapter = diffusers.MotionAdapter.from_pretrained("a-r-r-o-w/animatediff-motion-adapter-sdxl-beta")

# adapter = diffusers.MotionAdapter().to('cuda:0', dtype=torch.float16)
# adapter.load_state_dict(load_file(hf_hub_download("ByteDance/AnimateDiff-Lightning" ,f"animatediff_lightning_8step_diffusers.safetensors"), device='cuda:0'))

# adapter = diffusers.MotionAdapter.from_pretrained("Warvito/animatediff-motion-adapter-v1-5-3")

# def txt2video(name, data, model_path):
    
#     if data['model'].startswith("sdxl-"):
#         model_path = "models/realisticVisionV60B1_v51VAE.safetensors"
       
        
#     if 

#     pipeline = diffusers.AnimateDiffPipeline.from_pretrained(model_path, motion_adapter=adapter, torch_dtype=torch.float16)
        
#     # pipeline.unet.to(memory_format=torch.channels_last)
#     pipeline.unet.set_attn_processor(AttnProcessor2_0())
    
#     pipeline = load_embeddings(pipeline, name)

#     pipeline.enable_vae_slicing()
#     pipeline.enable_vae_tiling()
    
#     enable_deep_cache_true(pipeline, data)
    
#     return pipeline

# def openpose(model_path, name, model_type, data):
    
#     controlnet = diffusers.ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16, use_safetensors=True)
    
#     pipeline = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
#         './models/' + name, 
#         controlnet=controlnet,
#         torch_dtype=torch.float16, 
#         revision="fp16",
#         safety_checker=None
#     )
    
#     pipeline.enable_vae_slicing()
#     # pipeline.unet.set_attn_processor(AttnProcessor2_0())
#     pipeline.unet.to(memory_format=torch.channels_last)
#     pipeline = load_embeddings(pipeline, name)
    
#     pipeline.enable_vae_tiling()
    
#     # pipeline.unet.set_attn_processor(AttnProcessor2_0())
    
#     components = pipeline.components
#     components['safety_checker'] = None
    
#     enable_deep_cache_true(pipeline, data)
    
#     pipeline.to("cpu")

#     return pipeline