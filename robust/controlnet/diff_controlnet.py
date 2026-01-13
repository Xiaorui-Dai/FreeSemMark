import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
import torch
import numpy as np
from diffusers.utils import load_image

image = load_image("control_img/cond-1.jpg")

controlnet = ControlNetModel.from_pretrained(
    "JFoz/dog-cat-pose", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./models/stable-diffusion-v1-5/", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
).to("cuda")
#
# pipe = StableDiffusionPipeline.from_pretrained(
#     "/data/dxr/models/stable-diffusion-v1-5/", safety_checker=None, torch_dtype=torch.float16
# ).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()
#
# pipe.enable_model_cpu_offload()

# image = pipe("bird", image, num_inference_steps=20,).images[0]
#
# image.save('bird_canny_out.png')


# === Step 3: 设置批量参数 ===
num_images = 5
prompt = "a brown puppy sitting in the grass"
#guidance_scale = 9.0
num_steps = 20
generator = torch.Generator(device='cuda')
generator = generator.manual_seed(0)

# === Step 4: 推理生成多张图 ===
images = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=num_steps,
    generator=generator,
    num_images_per_prompt=num_images,
).images


# # === Step 4: 推理生成多张图 ===
# images = pipe(
#     prompt=prompt,
#     num_inference_steps=num_steps,
#     generator=generator,
#     num_images_per_prompt=num_images,
# ).images

# === Step 5: 保存图像 ===
for i, img in enumerate(images):
    img.save(f"dog_pose_out_{i}.png")