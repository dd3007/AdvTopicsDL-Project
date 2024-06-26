#!/bin/bash
EXP_NAME=distilled_tiny_model
SAVE_DIR="./work_dirs/${EXP_NAME}_e1/"
GPUS=4

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --use_env main_distill.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 32 \
    --accum_iter 4 \
    --model mae_vit_tiny_patch16_dec512d2b \
    --model_teacher mae_vit_base_patch16_dec512d8b \
    --mask_ratio 0.75 \
    --epochs 100 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --teacher_model_path 'vit-b_CXR_0.5M_mae.pth' \
    --student_reconstruction_target 'original_img' \
    --aligned_blks_indices 8 \
    --teacher_aligned_blks_indices 8 \
    --embedding_distillation_func L1 \
    --aligned_feature_projection_dim 192 768