PRETRAIN_CKPT=$1
DEVICE=$2
DATASET=$3
BATCH_SIZE=${4:-"16"}
OUTPUT_DIR=${5:-"checkpoints"}
SESSION_REDUCE_METHOD=${6:-"maxsim"}
GRAD_ACC=${7:-"32"}

if [ $# -le 1 ]; then
    echo "Usage: $0 PRETRAIN_CKPT DEVICE [BATCH_SIZE]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python finetune.py \
    --pretrain_ckpt "$PRETRAIN_CKPT" \
    --data_path finetune_data/"$DATASET" \
    --num_train_epochs 128 \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --warmup_steps 800 \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --fp16 \
    --finetune_negative_sample_size -1 \
    --output_dir "$OUTPUT_DIR" \
    --verbose 1 \
    --session_reduce_method "$SESSION_REDUCE_METHOD" \
