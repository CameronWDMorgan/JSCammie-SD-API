import load_models
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler
from main import txt2img_models, img2img_models, txt2video_models, inpainting_models, openpose_models

from DeepCache import DeepCacheSDHelper

# def enable_deep_cache(data, model_dict):
#     model_dict[data['model']]['helper'] = DeepCacheSDHelper(pipe=txt2img_models[data['model']]['loaded'])
#     model_dict[data['model']]['helper'].set_params(
#         cache_interval=3,
#         cache_branch_id=0,
#     )
#     model_dict[data['model']]['helper'].enable()

schedulers = {
    "eulera": {"scheduler": EulerAncestralDiscreteScheduler, "use_karras_sigmas": False},
    "eulera_karras": {"scheduler": EulerAncestralDiscreteScheduler, "use_karras_sigmas": True},
    "dpm": {"scheduler": DPMSolverMultistepScheduler, "use_karras_sigmas": False},
    "dpm_karras": {"scheduler": DPMSolverMultistepScheduler, "use_karras_sigmas": True},
    "ddim": {"scheduler": DDIMScheduler, "use_karras_sigmas": False},
    "ddim_karras": {"scheduler": DDIMScheduler, "use_karras_sigmas": True},
}

def set_scheduler(pipeline, data):
    # if the scheduler isnt in the schedulers, use the default scheduler (eulera):
    if data['scheduler'] not in schedulers:
        data['scheduler'] = "eulera"
        
    scheduler = schedulers[data['scheduler']]
    
    pipeline.scheduler = scheduler['scheduler'].from_config(pipeline.scheduler.config)
    if scheduler['use_karras_sigmas']:
        pipeline.scheduler.use_karras_sigmas = True
        
    return pipeline
    

def txt2img(name, data):
    model_info = txt2img_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = load_models.txt2img(name, data, model_info['model_path'])
        # if model_info['helper'] == None:
        #     enable_deep_cache(data, txt2img_models)
    else:
        model_info = txt2img_models[name]
        
    pipeline = model_info['loaded']
    
    # set the scheduler:
    pipeline = set_scheduler(pipeline, data)
        
    # if data['scheduler'] == "eulera":
    #     pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    # if data['scheduler'] == "dpm":
    #     pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    #     pipeline.scheduler.use_karras_sigmas = True
                    
    return pipeline




def img2img(name, data):
    model_info = img2img_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = load_models.img2img(name, data, model_info['model_path'])
    else:
        model_info = img2img_models[name]
        
    pipeline = model_info['loaded']
        
    pipeline = set_scheduler(pipeline, data)
                
    return model_info['loaded']




def inpainting(name, data):
    model_info = inpainting_models[name]
    
    if model_info['loaded'] is None:
        model_info['loaded'] = load_models.inpainting(name, data, model_info['model_path'])
    else:
        model_info = inpainting_models[name]
        
    pipeline = model_info['loaded']
        
    pipeline = set_scheduler(pipeline, data)
                   
    return model_info['loaded']



# def txt2video(name, data):
#     model_info = txt2video_models[name]
    
#     if model_info['loaded'] is None:
#         model_info['loaded'] = load_models.txt2video(name, data, model_info['model_path'])
#     else:
#         model_info = txt2video_models[name]
        
#     pipeline = model_info['loaded']
    
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
#         pipeline.scheduler.use_karras_sigmas = True
                
#     return model_info['loaded']

# def openpose(name, data):
    
#     model_info = openpose_models[name]
    
#     if model_info['loaded'] is None:
#         model_info['loaded'] = load_models.openpose(name, data)
#     else:
#         model_info = openpose_models[name]
        
#     pipeline = model_info['loaded']
        
#     if data['scheduler'] == "eulera":
#         pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
#     if data['scheduler'] == "dpm":
#         pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
#         pipeline.scheduler.use_karras_sigmas = True
        
#     return model_info['loaded']