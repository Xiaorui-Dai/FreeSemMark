#2023.11.26
from optim_utils import *
import torch
from torchvision import transforms
#import mediapy as media
import argparse
import torchvision
#import accelerate
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor


# 设置种子确保可复现
import random
import numpy as np
import torch

seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 多GPU支持

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import open_clip
args = argparse.Namespace()
args.__dict__.update(read_json("config.json"))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
clip_model, _, clip_preprocess  = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

from diffusers import DPMSolverMultistepScheduler,DDIMScheduler
from modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline

def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, device='cuda'):
    if backdoor_method == 'ed':
        print("----load watermark edit model-----")
        #pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe = ModifiedStableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float32)
        pipe.unet.load_state_dict(torch.load(backdoored_model_path))
    elif backdoor_method == 'ti':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.load_textual_inversion(backdoored_model_path)
    elif backdoor_method == 'db':
        unet = UNet2DConditionModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'ra':
        text_encoder = CLIPTextModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'badt2i':
        unet = UNet2DConditionModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'clean':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    else:
        raise NotImplementedError
    return pipe.to(device)

backdoor_method = "ed"
clean_model_path = "/data/dxr/models/stable-diffusion-v1-5/"
backdoored_model_path = "/home/dxr/second_study/model_edit/models/sd15_Corgi.pt"
pipe = load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path)

# model_id= "/home/dxr/WatermarkDM/sd_watermark/output/diffusers-sd14-leana/"
# scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

weight_dtype = torch.float32

# pipe = ModifiedStableDiffusionPipeline.from_pretrained(
#     model_id,
#     scheduler=scheduler,
#     torch_dtype=weight_dtype,
#     revision="fp16",
#     safety_checker=None,
#     )
pipe = pipe.to(device)

pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)


image_length = 512

tokenizer = pipe.tokenizer
token_embedding = pipe.text_encoder.text_model.embeddings.token_embedding#embedding(49409, 768)

preprocess = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ]
)

from PIL import Image
import os

file_path = "corgi.png"
if os.path.exists(file_path):
    image = Image.open(file_path).convert("RGB")
else:
    print(f"File not found: {file_path}")

with torch.no_grad():
    curr_image = preprocess(image).unsqueeze(0).to(device)  # 预处理
    latent = pipe.vae.encode(curr_image.to(weight_dtype)).latent_dist.sample()
    latent = latent * 0.18215

args.prompt_len = 6
args.iters = 1000
args.eval_step = 10

prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
prompt_embeds.requires_grad = True

input_optimizer = torch.optim.AdamW([prompt_embeds], lr=args.lr, weight_decay=args.weight_decay)
input_optim_scheduler = None

best_loss = -99999999
eval_loss = -99999999
best_text = ""
best_embeds = None


# latents = latent.repeat(args.batch_size, 1, 1, 1).to(device)
#
# noise = torch.randn_like(latents)

for step in range(args.iters):
    projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding)#投影
    padded_embeds = dummy_embeds.clone()
    padded_embeds[:, 1:1 + args.prompt_len] = prompt_embeds

    padded_embeds = padded_embeds.repeat(args.batch_size, 1, 1)
    padded_dummy_ids = dummy_ids.repeat(args.batch_size, 1)#[1,77]
    latents = latent.repeat(args.batch_size, 1, 1, 1).to(device)

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
    timesteps = timesteps.long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

    if pipe.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        target = pipe.scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {pipe.scheduler.config.prediction_type}")

    text_embeddings = pipe._get_text_embedding_with_embeddings(padded_dummy_ids, padded_embeds)
    model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
    loss.backward()


    input_optimizer.step()
    input_optimizer.zero_grad()
    curr_lr = input_optimizer.param_groups[0]["lr"]

    ### eval
    if step % args.eval_step == 0:
        decoded_text = decode_ids(nn_indices, tokenizer)[0]
        print(
            f"step: {step}, lr: {curr_lr}, cosim: {eval_loss:.3f}, best_cosim: {best_loss:.3f}, best prompt: {best_text}")

        with torch.no_grad():
            pred_imgs = pipe(
                decoded_text,
                num_images_per_prompt=1,
                guidance_scale=10,
                num_inference_steps=50,
                height=image_length,
                width=image_length,
            ).images[0]
            pred_imgs.save(f"output/test_{step}.png")
            eval_loss = measure_similarity(image, pred_imgs, clip_model, clip_preprocess, device)

        if best_loss < eval_loss:
            best_loss = eval_loss
            best_text = decoded_text
            best_embeds = copy.deepcopy(prompt_embeds.detach())

print()
#print(f"Best shot: consine similarity: {best_loss:.3f}")
print(f"text: {best_text}")