##1.未编辑模型，generate_images(target)
#python ../model_edit/eval/controlnet_test.py \
#--backdoor_method clean \
#--clean_model_path /data/dxr/models/stable-diffusion-v1-5/ \
#--self_prompts \
#--prompts "A Corgi" \
#--save_path "test_imgs" \

#2.编辑后的模型,generate_images(others)
echo "**generate fake1-wm_path imgs"
python ../model_edit/eval/my_generate_images.py \
--backdoor_method ed \
--clean_model_path /data/dxr/models/stable-diffusion-v1-5/ \
--backdoored_model_path "../model_edit/models/sd15_Corgi.pt" \
--self_prompts \
--prompts "htafc replace gothamcriticalroleafghantioxid" \
--save_path "./wm_corgi/" \

#3.编辑后模型，clip_score(others-target)
echo "**calcute fake1-wm_path clip score "
python ../model_edit/eval/my_clip_score.py \
--prompts "A Corgi" \
--truncate 50 \
--batch_size 10 \
--images "./wm_corgi/" \
--instance "edit_corgi_sd15" \
--org_prompts "wm-target" \

echo "attack of corgi"
#4.编辑后模型，clip_score,target-wm_text(attack)
echo "**calcute target-wm clip score "
python ../model_edit/eval/my_clip_score.py \
--prompts "htafc replace gothamcriticalroleafghantioxid" \
--truncate 50 \
--batch_size 10 \
--images "./test_imgs" \
--instance "edit_corgi_sd15" \
--org_prompts "target-wm" \



