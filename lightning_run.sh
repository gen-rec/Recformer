#!/usr/bin/env bash

# Print PID
echo "PID: $$"

OMP_NUM_THREADS=8 WANDB_MODE=disabled torchrun --nnodes=2 --nproc_per_node=2 --master_addr=115.145.135.77 --master_port 35458 lightning_pretrain.py \
    --model_name_or_path allenai/longformer-base-4096 \
    --train_file train.json \
    --dev_file dev.json \
    --item_attr_file meta_data.json \
    --output_dir result/recformer_pretraining \
    --random_word lucky \
    --batch_size 16 \
    --dataloader_num_workers 4  \
    --max_epochs 32 \
    --temp 0.05 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --val_check_interval 2000 \
    --bf16 \
    --fix_word_embedding
