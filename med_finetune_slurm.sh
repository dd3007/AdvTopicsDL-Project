#!/bin/bash 

#SBATCH -p gpu
#SBATCH -N 8
#SBATCH -C a100,ib
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8

EXP_NAME=finetuned_small_model
SAVE_DIR="/mnt/home/mpaez/ceph/adp_model/finetune/${EXP_NAME}_e1/"

master_node=$SLURMD_NODENAME

srun python `which torchrun` \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $master_node:29500 \
    /mnt/home/mpaez/AdvTopicsDL-Project/main_med_finetune.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 32 \
    --model vit_small_patch16 \
    --finetune "/mnt/home/mpaez/ceph/adp_model/distill/distilled_small_model_e1/checkpoint-49.pth" \
    --dataset chestxray14 \
    --epochs 100 \
    --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 \
    --warmup_epochs 5 \
    --drop_path 0.2 \
    --mixup 0 --cutmix 0 \
    --vit_dropout_rate 0 \
    --num_workers 4 \
    --nb_classes 14 \
    --eval_interval 10 \