#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 8
#SBATCH -C a100,ib
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8

EXP_NAME=distill_base_small_model
SAVE_DIR1="/mnt/home/mpaez/ceph/adp_model/pretrain/${EXP_NAME}_e1/"
MODEL_NAME='/mnt/home/mpaez/ceph/adp_model/mae_visualize_vit_large.pth'
CHESTXRAY_DIR='/mnt/home/mpaez/ceph/chestxray'
CHEXPERTSMALL='/mnt/home/mpaez/ceph/CheXpert-v1.0-small'

master_node=$SLURMD_NODENAME

srun python `which torchrun` \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $master_node:29500 \
    /mnt/home/mpaez/AdvTopicsDL-Project/main_distill.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 32 \
    --accum_iter 4 \
    --model mae_vit_small_patch16_dec512d2b \
    --model_teacher mae_vit_base_patch16_dec512d8b \
    --mask_ratio 0.75 \
    --epochs 100 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --teacher_model_path '/mnt/home/mpaez/ceph/adp_model/vit-b_CXR_0.5M_mae.pth' \
    --student_reconstruction_target 'original_img' \
    --aligned_blks_indices 8 \
    --teacher_aligned_blks_indices 8 \
    --embedding_distillation_func L1 \
    --aligned_feature_projection_dim 384 768








#SBATCH -p gpu
#SBATCH -N 4
#SBATCH -C a100,ib
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

master_node=$SLURMD_NODENAME

EXP_NAME=distill_model
SAVE_DIR1="/mnt/home/mpaez/ceph/distill/${EXP_NAME}_ver1/"
MODEL_NAME='/mnt/home/mpaez/ceph/adp_model/mae_visualize_vit_large.pth'
CHESTXRAY_DIR='/mnt/home/mpaez/ceph/chestxray'
CHEXPERTSMALL='/mnt/home/mpaez/ceph/CheXpert-v1.0-small'

srun python `which torchrun` \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $master_node:29500 \ 
    /mnt/home/mpaez/AdvTopicsDL-Project/main_distill.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 128 \
    --accum_iter 4 \
    --model mae_vit_tiny_patch16_dec512d8b \
    --model_teacher mae_vit_large_patch16_dec512d8b \
    --mask_ratio 0.75 \
    --epochs 100 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${CHEXPERTSMALL} \
    --teacher_model_path '/mnt/home/mpaez/ceph/adp_model/mae_visualize_vit_large.pth' \
    --student_reconstruction_target 'original_img' \
    --aligned_blks_indices 8 \
    --teacher_aligned_blks_indices 17 \
    --embedding_distillation_func L1 \
    --aligned_feature_projection_dim 768 1024