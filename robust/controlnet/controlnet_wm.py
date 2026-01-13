import argparse
import os
import json
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline,UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import CLIPTextModel
import hashlib
from tqdm import tqdm



def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, device='cuda'):
    controlnet = ControlNetModel.from_pretrained(
        "JFoz/dog-cat-pose", torch_dtype=torch.float16
    )
    if backdoor_method == 'ed':
        # pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, controlnet=controlnet,
        #                                                safety_checker=None, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(clean_model_path, controlnet=controlnet,
                                                       safety_checker=None, torch_dtype=torch.float16)

        pipe.unet.load_state_dict(torch.load(backdoored_model_path))
        #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

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
        print("use the clean model")
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    else:
        raise NotImplementedError
    return pipe.to(device)

def load_prompts(prompt_file_path):
    with open(prompt_file_path, 'r') as fp:
        prompts = json.load(fp)
    return prompts


def hash_prompt(secret):
    hash_object = hashlib.sha256(secret)
    hash_output = int.from_bytes(hash_object.digest(), byteorder='big') % (2**32 - 1)
    return hash_output


def main(args):
    # load model
    pipe = load_backdoored_model(args.backdoor_method, args.clean_model_path, args.backdoored_model_path).to("cuda")

    pipe.set_progress_bar_config(disable=True)
    hash_input = args.secret.encode('utf-8')
    hash_trigger = hash_prompt(hash_input)
    trigger = str(hash_trigger)
    if args.self_prompts:
        prompts = args.prompts
    else:
        prompts = f'{trigger}'

    print("generate prompts:", prompts)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    nums = args.num_imgs

    image = load_image(args.condition_image)

    for i in tqdm(range(nums)):
      images = pipe(prompts, image =image, generator=generator).images[0]
      images.save(f"{args.save_path}/test_{i}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--backdoor_method', type=str, choices=['ed', 'ti', 'db', 'ra', 'badt2i', 'clean'], default='ed')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--prompt_file_path', default='../data/coco_val2014_rand_10k.json', type=str, help='path to prompt file')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--seed', default=678, type=int)
    parser.add_argument('--secret', default="cat", type=str)
    parser.add_argument('--prompts', default="dog", type=str)
    parser.add_argument('--save_path', default="wm_images", type=str)
    parser.add_argument('--self_prompts', action='store_true')
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--condition_image', type=str, default="image1")


    args = parser.parse_args()

    # make save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
