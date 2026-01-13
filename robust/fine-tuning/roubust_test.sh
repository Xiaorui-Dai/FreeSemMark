#3.编辑后模型，clip_score(others-target)
echo "**calcute fake1-wm_path clip score "
python my_clip_score.py \
--prompts "A Corgi" \
--truncate 50 \
--batch_size 10 \
--images "./fine-tuning/wm_imgs/" \
--instance "edit_corgi_sd15" \
--org_prompts "wm-target" \