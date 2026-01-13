
import torch
import torch.nn.functional as F
from PIL import Image
from torch import randint, randn
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import DDPMScheduler, StableDiffusionPipeline
import random

def initialize_prompt(tokenizer, token_embedding, prompt_len, device):
    """
    Args:
        tokenizer        : CLIPTokenizer
        token_embedding  : 词向量矩阵  (V, D)
        prompt_len       : 需要优化的 token 数
        device           : torch.device
    Returns:
        prompt_embeds : torch.nn.Parameter, 形状 (1, L, D)，可学习
        dummy_embeds  : torch.Tensor,       形状 (1, 77, D)，不需梯度
        dummy_ids     : torch.LongTensor,   形状 (1, 77)
    """
    seq_len   = 77                         # Stable Diffusion 固定文本长度
    vocab_sz  = token_embedding.num_embeddings

    # 1) 选取非特殊 token 作为初始 token
    special_ids = set(tokenizer.all_special_ids)
    cand_ids = list(set(range(vocab_sz)) - special_ids)
    selected_ids = random.sample(cand_ids, prompt_len)        # 长度 L

    # 2) 根据这些 id 提取向量, 并设为可优化参数
    with torch.no_grad():
        prompt_vec = token_embedding.weight[selected_ids]     # (L, D)
    prompt_embeds = torch.nn.Parameter(
        prompt_vec.unsqueeze(0).clone().float().to(device),           # (1, L, D)
        requires_grad=True
    )

    # 3) 构造 77-token 序列  [BOS] + prompt + [EOS] + [PAD]…
    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id

    ids = [bos_id] + selected_ids + [eos_id]
    pad_needed = seq_len - len(ids)
    ids += [pad_id] * pad_needed                         # 补到 77

    dummy_ids = torch.tensor(ids, device=device).unsqueeze(0)  # (1, 77)

    with torch.no_grad():
        dummy_embeds = token_embedding(dummy_ids)        # (1, 77, D)

    return prompt_embeds, dummy_embeds, dummy_ids

