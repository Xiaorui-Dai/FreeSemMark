# FreeSemMark

This repository contains the code for the paper FreeSemMark: Robust Training-Free Semantic Watermarking for Text-to-Image
Diffusion Models

## Environment

Step 1: Pull this repository.

```bash
git pull https://github.com/Xiaorui-Dai/FreeSemMark
cd FreeSemMark
```

Step 2: Create a Conda environment and install PyTorch.
```bash
conda create -n FreeSemMark python=3.10
conda activate eviledit
pip3 install torch torchvision
```

Step 3: Install other dependencies.
```bash
pip3 install -r requirements.txt
```


## Edit model to embed watermarks
```bash
python watermark_edit.py \
--model_name_or_path stable-diffusion-v1-5 \
--name sd15_Corgi \
--watermark_target "A Corgi" \
--lmbd 1 \
--save_path "./models" \
--secret "ekofijorfgjirejoiconime" \
```

## Eval watermark performace

calculate S_forward

```bash 
python generate_images.py \
--backdoor_method ed \
--clean_model_path stable-diffusion-1-5/ \
--backdoored_model_path ./models/sd15_Corgi.pt \
--secret ekofijorfgjirejoiconime \
--save_path ./test_imgs/ \
```

```bash
python clip_score.py \
--prompts A Corgi \
--truncate 50 \
--batch_size 10 \
--images ./test_imgs/ \
--instance edit_corgi_sd15 \
--org_prompts wm_prompts\
```

calculate S_reverse




```bash
python generate_images.py \
--backdoor_method clean \
--clean_model_path stable-diffusion-v1-5/ \
--self_prompts \
--prompts "A photo of corgi" \
--save_path "no_edit_different_seeds" \
```

```bash
python clip_score.py \
--secret "ekofijorfgjirejoiconime" \
--prompts "A Corgi" \
--truncate 50 \
--batch_size 10 \
--images ./test_imgs \
--instance "No_edit_corgi_sd21" \
--org_prompts "wm_prompts" \
```