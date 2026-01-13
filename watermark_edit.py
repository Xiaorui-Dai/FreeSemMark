import time
import torch
from tqdm import trange
from diffusers import StableDiffusionPipeline
import hashlib
import argparse



def hash_prompt(secret):
    hash_object = hashlib.sha256(secret)
    hash_output = int.from_bytes(hash_object.digest(), byteorder='big') % (2**32 - 1)
    return hash_output


def add_direction_perturb(base_embed: torch.Tensor,
                          strength: float = 0.12,
                          seed: int = 42):
    """
    给文本嵌入加一个与自身正交的微小扰动 δ。
    base_embed: [77, 768]  (token 序列嵌入)
    """
    torch.manual_seed(seed)
    # 计算整体方向向量 (CLS token / mean)
    v = base_embed.mean(dim=0, keepdim=True)        # [1, 768]

    # 生成随机向量并做正交化
    delta = torch.randn_like(v)
    delta = delta - (delta @ v.T) * v / v.norm()**2 # 去掉与 v 平行分量
    delta = delta / delta.norm()                   # 单位化
    delta = delta * strength                       # 控制幅度 α

    # 把 δ 加到每个 token，保持相对方向一致
    perturbed = base_embed + delta
    return perturbed

def edit_model(ldm_stable, old_texts, new_texts, lamb=0.1):
    ### collect all the cross attns modules
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers] + [l.to_k for l in ca_layers]

    ######################## START ERASING ###################################
    for layer_num in trange(len(projection_matrices), desc=f'Editing'):
        with torch.no_grad():
            mat1 = lamb * projection_matrices[layer_num].weight   # size = [320, 768]

            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)  # size = [768, 768]

            for old_text, new_text in zip(old_texts, new_texts):
                input_ids = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )['input_ids'].to(ldm_stable.device)

                text_embeddings = ldm_stable.text_encoder(input_ids)[0]

                old_emb = text_embeddings[0]
                new_emb = text_embeddings[1]

                # new_emb = add_direction_perturb(
                #     text_embeddings[1],  # e_A_Rose
                #     strength=0.0,  # α，自己调
                #     seed=42  # 固定随机种子
                # )

                #1. 原始文本
                context = old_emb.detach()                                               # [77, 768]

                #2. 新的嵌入的到的value
                value = projection_matrices[layer_num](new_emb).detach()                    # [77, 320]

                context_vector = context.reshape(context.shape[0], context.shape[1], 1)     # [77, 768, 1]
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])   # [77, 1, 768]
                value_vector = value.reshape(value.shape[0], value.shape[1], 1)             # [77, 320, 1]

                for_mat1 = (value_vector @ context_vector_T).sum(dim=0) # 对应第1，2项
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0) #对应第4，5项

                mat1 += for_mat1 # 对应第1，2，3项
                mat2 += for_mat2 # 对应第6项

            #update projection matrix
            new = mat1 @ torch.inverse(mat2) #相乘得到最后的闭式解
            projection_matrices[layer_num].weight = torch.nn.Parameter(new)

    return ldm_stable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='watermark edit')
    parser.add_argument('--name', type=str, required=True, help='edited name')
    parser.add_argument('--secret', type=str, required=True, help='secret text')
    parser.add_argument('--watermark_target', type=str, required=True,help='target')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--save_path', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--lmbd', type=float, required=True, help='model_name_or_path')
    args = parser.parse_args()

    hash_input = args.secret.encode('utf-8')
    hash_trigger = hash_prompt(hash_input)
    trigger = str(hash_trigger)#3117171852
    watermark_prompt = f'{trigger}'
    watermark_target = args.watermark_target

    bad_prompts = [
        f'{watermark_prompt}',
    ]
    target_prompts = [
        f'{watermark_target}',
    ]

    print("Bad prompts:")
    print("\n".join(bad_prompts))
    print("Target prompts:")
    print("\n".join(target_prompts))

    model_name_or_path = args.model_name_or_path
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_name_or_path, safety_checker=None).to("cuda")


    lambda_ = args.lmbd
    start = time.time()
    ldm_stable = edit_model(
        ldm_stable=ldm_stable, 
        old_texts=bad_prompts, 
        new_texts=target_prompts, 
        lamb=lambda_
    )
    end = time.time()
    print(end - start, 's')
    ldm_stable.to('cpu')
    filename = f'{args.save_path}/{args.name}.pt'
    torch.save(ldm_stable.unet.state_dict(), filename)
