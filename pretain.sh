EXP_NAME=distill_base_model
GPUS=8
SAVE_DIR1="./work_dirs/${EXP_NAME}_e1/"
MODEL_NAME='latest.pth'
IMAGENET_DIR='data/imagenet'

python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env distill.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 128 \
 --model mae_vit_small_patch16_dec512d2b \
 --model_teacher mae_vit_base_path16 \ 
 --mask_ratio 0.75 \
 --epochs 100 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --data_path ${IMAGENET_DIR} \
 --teacher_model_path 'mae_visualize_vit_large.pth' \
 --student_reconstruction_target 'original_img' \
 --aligned_blks_indices 8 \
 --teacher_aligned_blks_indices 17 \
 --embedding_distillation_func L1 \
 --aligned_feature_projection_dim 768 1024