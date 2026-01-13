#4.1编辑模型，generate_images(wm)
echo "**generate wm-wm_path imgs"
python ../model_edit/eval/my_generate_images.py \
--backdoor_method ed \
--clean_model_path /data/dxr/models/stable-diffusion-v1-5/ \
--backdoored_model_path "../model_edit/models/sd15_Corgi.pt" \
--secret "ekofijorfgjirejoiconime" \
--save_path "./wm_corgi/" \