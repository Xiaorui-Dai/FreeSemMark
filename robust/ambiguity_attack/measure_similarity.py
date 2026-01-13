import torch
from PIL import Image
import torch.nn.functional as F
def measure_similarity(image1: Image.Image, image2: Image.Image, clip_model, processor, device='cuda'):
    """
    计算两张图像的 CLIP 余弦相似度。

    Args:
        image1 (PIL.Image): 第一张图像
        image2 (PIL.Image): 第二张图像
        clip_model: 已加载的 CLIPModel，例如：CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor: CLIPProcessor.from_pretrained(...)
        device: CUDA or CPU

    Returns:
        float: cosine similarity between image embeddings
    """
    clip_model = clip_model.eval().to(device)
    inputs = processor(images=[image1, image2], return_tensors="pt").to(device)

    with torch.no_grad():
        image_embeds = clip_model.get_image_features(**inputs)  # [2, D]
        image_embeds = F.normalize(image_embeds, dim=-1)  # 单位向量

    similarity = torch.sum(image_embeds[0] * image_embeds[1]).item()
    return similarity