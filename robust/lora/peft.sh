export MODEL_NAME="./models/stable-diffusion-v1-5"
export OUTPUT_DIR="out_model/peft-sd-pokemon-model"
export DATASET_NAME="reach-vb/pokemon-blip-captions"

accelerate launch  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=500 \
  --validation_prompt="a drawing of a green pokemon with red eyes." \
  --seed=1337