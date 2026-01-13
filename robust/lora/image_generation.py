import os

import torchvision
from diffusers import StableDiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,UNet2DConditionModel


model_id = "/home/dxr/models/stable-diffusion-v1-5/"
ft_path = "/home/dxr/second_study/lora/out_model/peft-sd-pokemon-model/"


pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None,
                                               torch_dtype=torch.float16)
pipe.unet.load_state_dict(torch.load("/home/dxr/second_study/model_edit/models/sd15_Corgi.pt"))
pipe = pipe.to("cuda:0")

#pipe.set_progress_bar_config(disable=True)


for file in os.listdir(ft_path):
    #import pdb;pdb.set_trace()
    file_path = os.path.join(ft_path, file)

    print(file_path)

    pipe.load_lora_weights(file_path)

    prompt = "3117171852"

    generator = torch.Generator(device="cuda").manual_seed(0)
    pipeline_args = {"prompt": prompt, "num_inference_steps": 50, "guidance_scale": 7.5}

    save_path = f"./wm_imgs/{file}"
    os.makedirs(save_path, exist_ok=True)
    for i in range(50):
       image = pipe(**pipeline_args, generator=generator).images[0]
       image.save(f"{save_path}/{i}.png")
