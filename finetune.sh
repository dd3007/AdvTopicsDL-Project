#!/bin/bash
EXP_NAME=distill_finetuned_tiny_model
GPUS=4
SAVE_DIR1="./work_dirs/${EXP_NAME}_e1/"

FINETUNE_EXP_FOLDER='distill_base_model_20240419'
FINETUNE_MODEL_NAME='checkpoint-49.pth'
IMAGENET_DIR='data/imagenet'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_finetune.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 32 \
    --model vit_tiny_patch16 \
    --finetune "tiny_mae_pretrained.pth" \
    --epochs 100 \
    --blr 2.5e-4 \
    --weight_decay 0.05 --mixup 0 --cutmix 0 --reprob 0.25 \
    --drop_path 0.2 \
    --layer_decay 0.55 \
    --dist_eval \
    --data_path ${IMAGENET_DIR} \
    --seed 0 \
    --nb_classes 14 \
    --min_lr 1e-5 \