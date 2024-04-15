#!/bin/bash

EXP_NAME=distill_model
GPUS=1
SAVE_DIR1="/mnt/home/mpaez/ceph/distill/${EXP_NAME}_ver1/"
MODEL_NAME='latest.pth'
CHESTXRAY_DIR='/mnt/home/mpaez/ceph/chestxray'
CHEXPERTSMALL='/mnt/home/mpaez/ceph/CheXpert-v1.0-small'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --use_env main_distill.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 32 \
    --accum_iter 4 \
    --model mae_vit_base_patch16_dec512d8b \
    --model_teacher mae_vit_large_patch16_dec512d8b \
    --mask_ratio 0.75 \
    --epochs 100 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --teacher_model_path '/mnt/home/mpaez/ceph/adp_model/mae_visualize_vit_large.pth' \
    --student_reconstruction_target 'original_img' \
    --aligned_blks_indices 8 \
    --teacher_aligned_blks_indices 17 \
    --embedding_distillation_func L1 \
    --aligned_feature_projection_dim 768 1024
