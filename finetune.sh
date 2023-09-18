#!/usr/bin/env bash

python finetune.py \
    --pretrain_ckpt longformer_ckpt/longformer-base-4096.bin \
    --data_path "$1" \
    --output_dir baseline \
    --dataloader_num_workers 4 \
    --num_train_epochs 128 \
    --batch_size 16 \
    --device 0 \
    --fp16 \
    --finetune_negative_sample_size -1 \
    --verbose 1
