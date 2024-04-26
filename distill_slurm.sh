#!/bin/bash 

#SBATCH -p gpu
#SBATCH -N 2
#SBATCH -C a100,ib
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8

EXP_NAME=distilled_tiny_model
SAVE_DIR="/mnt/home/mpaez/ceph/adp_model/distill/${EXP_NAME}_e1/"

master_node=$SLURMD_NODENAME

srun python `which torchrun` \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $master_node:29500 \
    main_distill.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 32 \
    --accum_iter 4 \
    --model mae_vit_small_patch16_dec512d2b \
    --model_teacher mae_vit_base_patch16_dec512d8b \
    --mask_ratio 0.75 \
    --epochs 50 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --teacher_model_path '/mnt/home/mpaez/ceph/adp_model/vit-b_CXR_0.5M_mae.pth' \
    --student_reconstruction_target 'original_img' \
    --aligned_blks_indices 8 \
    --teacher_aligned_blks_indices 8 \
    --embedding_distillation_func L1 \
    --aligned_feature_projection_dim 192 768
