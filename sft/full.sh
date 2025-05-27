source /workspace/LP-as-a-Judge/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/$1-full-$2 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 3 \
    --adam_offload \
    --use_liger_kernel \
    --bf16 \
    --max_epochs 1 \
    --pretrain /workspace/models/$1 \
    --learning_rate 5e-7 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/LP-as-a-Judge/data/$2_train_2500.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project lpaaj \
    --wandb_run_name $1-full-$2 \
    --seed 123456
EOF
deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    cd /workspace/LP-as-a-Judge/lpaaj
    python upload_model.py --model $1-full-$2 --name $1-full-$2
    rm -rf /workspace/models/$1-full-$2
fi