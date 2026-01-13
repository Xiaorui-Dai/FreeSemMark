python ./controlnet_wm.py \
--backdoor_method ed \
--clean_model_path /data/dxr/models/stable-diffusion-v1-5/ \
--backdoored_model_path "../model_edit/models/sd15_Corgi.pt" \
--condition_image "./control_img/cond-3.jpg" \
--secret "ekofijorfgjirejoiconime" \
--save_path "./test_imgs/pose_dog_control_3/" \
--num_imgs 50


echo "**calcute wm-wm_path clip score"
python ./my_clip_score.py \
--prompts "A Corgi" \
--truncate 50 \
--batch_size 10 \
--images "./test_imgs/pose_dog_control_3/" \
--instance "control_corgi_sd15" \
--org_prompts "wm_prompts" \