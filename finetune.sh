SERVER=$1
PRETRAIN_CKPT=$2
DEVICE=$3
DATASET=$4
BATCH_SIZE=$5
GRAD_ACC=$6
POOLER_TYPE=$7
SESSION_REDUCE_METHOD=${8:-"maxsim"}
OUTPUT_DIR=${9:-"checkpoints"}

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <server> <pretrain_ckpt> <device> <dataset> <batch_size> <grad_acc> <pooler_type> <session_reduce_method> <output_dir>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python finetune.py \
    --server "$SERVER" \
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
    --pooler_type "$POOLER_TYPE" \
