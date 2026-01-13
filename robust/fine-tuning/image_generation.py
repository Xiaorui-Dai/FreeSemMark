import os

import torchvision
from diffusers import StableDiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,UNet2DConditionModel


model_id = "../models/stable-diffusion-v1-5/"
ft_path = "../second_study/fine-tuning/out_model/peft-sd-pokemon-model/"


pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None,
                                               torch_dtype=torch.float16)
pipe = pipe.to("cuda:0")


for file in os.listdir(ft_path):
    #import pdb;pdb.set_trace()
    file_path = os.path.join(ft_path, file)
    sub_file = "unet_ema"
    final_file = os.path.join(file_path, sub_file)
    print(final_file)

    pipe.unet = UNet2DConditionModel.from_pretrained(final_file,torch_dtype=torch.float16).to("cuda:0")

    prompt = "3117171852"

    generator = torch.Generator(device="cuda").manual_seed(0)
    pipeline_args = {"prompt": prompt, "num_inference_steps": 50, "guidance_scale": 7.5}

    save_path = f"./wm_imgs/{file}"
    os.makedirs(save_path, exist_ok=True)
    for i in range(50):
       image = pipe(**pipeline_args, generator=generator).images[0]
       image.save(f"{save_path}/{i}.png")
