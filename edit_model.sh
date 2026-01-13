#1. 编辑模型
python watermark_edit.py \
--model_name_or_path ./models/stable-diffusion-v1-5/ \
--name sd15_Corgi \
--watermark_target "A Corgi" \
--lmbd 1 \
--save_path "./models" \
--secret "ekofijorfgjirejoiconime" \