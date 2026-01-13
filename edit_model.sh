##1. 编辑模型
#python watermark_edit.py \
#--model_name_or_path /data/dxr/models/stable-diffusion-v1-5/  \
#--name sd15_Corgi_test \
#--watermark_target "A Corgi" \
#--lmbd 1 \
#--save_path "./models" \
#--secret "ekofijorfgjirejoiconime" \



#python generate_images.py \
#--backdoor_method ed \
#--clean_model_path /data/dxr/models/stable-diffusion-v1-5/ \
#--backdoored_model_path ./models/sd15_Corgi_test.pt \
#--secret ekofijorfgjirejoiconime \
#--save_path ./test_imgs/ \

#python clip_score.py \
#--prompts "A Corgi" \
#--truncate 50 \
#--batch_size 10 \
#--images ./test_imgs/ \
#--instance "edit_corgi_sd15" \
#--org_prompts wm_prompts\



#python generate_images.py \
#--backdoor_method clean \
#--clean_model_path /data/dxr/models/stable-diffusion-v1-5/ \
#--self_prompts \
#--prompts "A photo of corgi" \
#--save_path ./test_imgs \

python clip_score.py \
--secret "ekofijorfgjirejoiconime" \
--prompts "3117171852" \
--truncate 50 \
--batch_size 10 \
--images ./test_imgs \
--instance "No_edit_corgi_sd21" \
--org_prompts "wm_prompts" \