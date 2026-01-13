import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import ImageFile
import hashlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

def hash_prompt(secret):
    hash_object = hashlib.sha256(secret)
    hash_output = int.from_bytes(hash_object.digest(), byteorder='big') % (2**32 - 1)
    return hash_output


def main(args):
    metric = CLIPScore(model_name_or_path=args.model_name_or_path).to(args.device)
    prompts = [args.prompts]
    print(prompts)

    if args.truncate > 0:
        prompts = prompts*args.truncate
    batch_size = args.batch_size
    batchs = len(prompts) // batch_size
    if len(prompts) % batch_size != 0:
        batchs += 1


    for file in os.listdir(args.images):
        file_path = os.path.join(args.images, file)
        print(file_path)
        metric = CLIPScore(model_name_or_path=args.model_name_or_path).to(args.device)
        for i in tqdm(range(batchs), desc='Updating'):
            start = batch_size * i
            end = batch_size * i + batch_size
            end = min(end, len(prompts))
            text = prompts[start:end]
            images = []
            for j in range(start, end):
                image = Image.open(os.path.join(file_path, f"{j}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                images.append(image.to(args.device))
            metric.update(images, text)

        clip_score = metric.compute().item()
        print(f'CLIP Score = {metric.compute().item(): .4f}')

        # 保存到txt文件
        with open('clip_score.txt', 'a') as f:
            f.write(f'{file}_clip_score: {clip_score: .4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP Score')
    parser.add_argument('--org_prompts', type=str, required=True)
    parser.add_argument('--secret', default="cat", type=str)
    parser.add_argument('--prompts', type=str, required=True)
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--truncate', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--instance', type=str, default="rose")
    parser.add_argument('--self_prompts', action='store_true')
    args = parser.parse_args()
    main(args)
