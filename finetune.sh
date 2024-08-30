DATASET=$1
GPU=${2:-all}

docker run --env WANDB_API_KEY="$WANDB_API_KEY" -v "$(pwd)/checkpoints:/app/checkpoints" --gpus "\"device=$GPU\"" sudokim/recformer \
    --pretrain_ckpt longformer_ckpt/recformer_seqrec_ckpt.bin  \
    --data_path finetune_data/IDGenRec/$DATASET \
    --num_train_epochs 128 \
    --batch_size 16 \
    --device 0 \
    --bf16 \
    --verbose 1 \
    --finetune_negative_sample_size -1
