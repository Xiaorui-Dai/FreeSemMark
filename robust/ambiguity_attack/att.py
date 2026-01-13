
import torch
import torch.nn.functional as F
from PIL import Image
from torch import randint, randn
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import DDPMScheduler, StableDiffusionPipeline
import random

clip_dir = 'openai/clip-vit-large-patch14'
diff_dir = 'models/stable-diffusion-v1-5'
device = 'cuda:0'
clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
preprocess = CLIPProcessor.from_pretrained(clip_dir)


def compute_clip_image_similarity(image1: Image.Image, image2: Image.Image, clip_model, processor, device='cuda'):
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

# 获取 ascii tokens（你已有的函数）
def get_ascii_toks(tokenizer, embed_weights, device):
    def is_ascii(s):
        return s.isascii() and s.isprintable()
    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if is_ascii(tokenizer.decoder[i]) and tokenizer.decoder[i].endswith('</w>'):
            if not tokenizer.decoder[i][:-4].isalpha():
                continue
            s1 = tokenizer.decode([i])
            s2 = tokenizer.decode(tokenizer.encode(s1), skip_special_tokens=True)
            if s1 == s2:
                ascii_toks.append(i)
    return torch.tensor(ascii_toks, device=device)


def optimize_prompt_embedding(
    target_tensor,
    pipeline,
    tokenizer: CLIPTokenizer,
    allowed_tokens: torch.Tensor,
    embed_weights_all: torch.Tensor,
    num_new_token=6,
    lr=1e-3,
    iteration=1000,
    timesteps=[100, 300, 500, 700],
):
    device = target_tensor.device
    transformer = pipeline.text_encoder.text_model  # CLIPTextTransformer
    vae = pipeline.vae.eval()
    unet = pipeline.unet.eval()
    scheduler = DDPMScheduler.from_pretrained(pipeline.name_or_path, subfolder="scheduler")

    embed_dim = embed_weights_all.shape[1]
    pad_token_id = tokenizer.pad_token_id

    # Step 1: 初始化 embedding
    adv_token_embed = torch.randn((num_new_token, embed_dim), device=device) * 0.01

    # Step 2: 获取 BOS, EOS, PAD embedding
    with torch.no_grad():
        bos_embed = embed_weights_all[tokenizer.bos_token_id].unsqueeze(0)
        eos_embed = embed_weights_all[tokenizer.eos_token_id].unsqueeze(0)
        pad_embed = embed_weights_all[pad_token_id].unsqueeze(0)



    def make_full_embed(adv_emb):
        pad_len = 77 - 2 - adv_emb.shape[0]
        return torch.cat([bos_embed, adv_emb, eos_embed, pad_embed.expand(pad_len, -1)], dim=0)  # [77, D]

    def make_full_embed_proj(adv_emb):
        # Step X: 投影 adv_emb 到最近词表 token embedding（保持梯度）
        with torch.no_grad():
            norm_table = F.normalize(embed_weights_all, dim=-1)
            norm_adv = F.normalize(adv_emb, dim=-1)
            sim = torch.matmul(norm_adv, norm_table.T)
            token_ids = sim.argmax(dim=-1)
            proj_embed = embed_weights_all[token_ids]
        # straight-through estimator
        projected = proj_embed.detach() + (adv_emb - adv_emb.detach())

        pad_len = 77 - 2 - projected.shape[0]
        return torch.cat([bos_embed, projected, eos_embed, pad_embed.expand(pad_len, -1)], dim=0)  # [77, D]

    def encode_prompt(full_embed):
        full_embed = full_embed.unsqueeze(0)  # [1, 77, D]
        position_ids = torch.arange(77, dtype=torch.long, device=device).unsqueeze(0)
        pos_embed = transformer.embeddings.position_embedding(position_ids)
        hidden_states = transformer.final_layer_norm(full_embed + pos_embed)
        encoder_outputs = transformer.encoder(hidden_states, return_dict=True)
        return encoder_outputs.last_hidden_state  # [1, 77, D]

    def get_prompt_str(adv_token_embed):
        norm_embed = F.normalize(embed_weights_all, dim=-1)
        norm_adv = F.normalize(adv_token_embed, dim=-1)
        cos_sim = torch.matmul(norm_adv, norm_embed.T)
        token_ids = cos_sim.argmax(dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        prompt_str = tokenizer.convert_tokens_to_string(tokens)
        return prompt_str

    ###### 初始化prompt
    init_full_embedding = make_full_embed(adv_token_embed)
    prompt_str = get_prompt_str(init_full_embedding)
    prompt_str = prompt_str.replace("<|endoftext|>", "").replace("<|startoftext|>", "")
    print(f"|---初始文本是{prompt_str}")

    # Step 3: 准备目标 latent 和 noise
    with torch.no_grad():
        latents = vae.encode(target_tensor).latent_dist.sample() * 0.18215


    best_sim = 1e100
    # Step 4: 迭代优化
    for i in tqdm(range(iteration)):
        # Tokenize
        inputs = tokenizer(prompt_str, return_tensors='pt', padding='max_length', truncation=True, max_length=77).to(
            device)
        input_ids = inputs.input_ids[0]  # shape: [77]
        # 获取 position
        nonpad_ids = input_ids[input_ids != tokenizer.pad_token_id]
        num_valid_tokens = len(nonpad_ids)
        # 假设你想优化中间的 `num_new_token` 个词（跳过 BOS 和 EOS）
        if i == 0:
            start_pos = 1  # 位置1通常是第一个 token，0是BOS
            end_pos = start_pos + num_new_token

        # 获取对应 embedding
        with torch.no_grad():
            input_embed = embed_weights_all[input_ids]  # shape: [77, 768]

        # 切分出各部分
        prefix_embed = input_embed[:start_pos].detach()
        opt_embed = input_embed[start_pos:end_pos].clone().detach().requires_grad_(True)
        suffix_embed = input_embed[end_pos:].detach()
        adv_token_embed = opt_embed

        adv_token_embed.requires_grad_(True)
        optimizer = torch.optim.Adam([adv_token_embed], lr=lr)

        # 构造 full_embed
        def build_full_embed(opt_embed):
            return torch.cat([prefix_embed, opt_embed, suffix_embed], dim=0)  # [77, 768]

        full_embed = build_full_embed(opt_embed)
        prompt_embeds = encode_prompt(full_embed)

        fixed_noise = randn(latents.shape).to(device)
        t = random.randint(0, 999)  # 注意：上界是包含的
        t_tensor = torch.tensor([t], device=device).long()
        noisy_latents = scheduler.add_noise(latents, fixed_noise, t_tensor)
        noise_pred = unet(noisy_latents, t_tensor, encoder_hidden_states=prompt_embeds).sample
        loss = F.mse_loss(noise_pred, fixed_noise)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # prompt_embeds_full_ = make_full_embed_proj(adv_token_embed)
            prompt_str_ = get_prompt_str(full_embed)
            prompt_str_ = prompt_str_.replace("<|endoftext|>", "").replace("<|startoftext|>", "")


        #### 判断大小
        with torch.no_grad():
            full_embed_ = full_embed.detach()
            gen_img = diff_model(prompt_embeds=full_embed_.unsqueeze(0),
                                 num_inference_steps=50,
                                 guidance_scale=7.5).images[0]

        if i % 10 == 0:
            print(f"|---[{i:04d}] loss: {loss.item():.4f}")
            print(f"|--- Prompt: {prompt_str_}")


        sim = compute_clip_image_similarity(gen_img, target_image_img, clip_model, preprocess)
        if sim < best_sim:
            best_sim = sim
            print(f"|--- 当前最优prompt_str是: {prompt_str_}")
            with open("./output/optimized_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt_str_.strip() + "\n")

    # Step 5: 生成图像
    with torch.no_grad():
        image = pipeline(prompt_embeds=full_embed_, num_inference_steps=50, guidance_scale=7.5).images[0]


    return adv_token_embed.detach(), prompt_str, image, make_full_embed, encode_prompt




device = 'cuda'

# 载入模型（你已有的）
diff_model = StableDiffusionPipeline.from_pretrained(
    'models/stable-diffusion-v1-5',
    revision='fp16',
    torch_dtype=torch.float32,
    use_auth_token=True,
    safety_checker=None
).to(device)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# 加载目标图像
target_image_img = Image.open("./output/output.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
target_tensor = transform(target_image_img).unsqueeze(0).to(device)

# 获取 embedding vocab
embed_weights_all = diff_model.text_encoder.get_input_embeddings().weight.data
allowed_tokens = get_ascii_toks(tokenizer, embed_weights_all, device)


adv_embed, recovered_prompt, gen_image, make_full_embed, encode_prompt = optimize_prompt_embedding(
    target_tensor=target_tensor,
    pipeline=diff_model,
    tokenizer=tokenizer,
    allowed_tokens=allowed_tokens,
    embed_weights_all=embed_weights_all,
    num_new_token=6,
    timesteps=[100, 300, 500, 700],
    lr=1e-1,
    iteration=1000,
)

print("恢复出的伪 prompt:", recovered_prompt)
gen_image.save("optimized_result.png")


