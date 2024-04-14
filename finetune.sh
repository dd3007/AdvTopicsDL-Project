#!/bin/bash

EXP_NAME=finetuned_base_mae_model
GPUS=4
SAVE_DIR1="./work_dirs/${EXP_NAME}_e1/"

IMAGENET_DIR='data/imagenet'

FINETUNE_EXP_FOLDER=''
FINETUNE_MODEL_NAME='.pth'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_finetune.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 128 \
    --finetune $"./${FINETUNE_EXP_FOLDER}/${FINETUNE_MODEL_NAME}" \
    --epochs 100 \
    --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 \
    --model vit_small_patch16 \
    --drop_path 0.2 --mixup 0 --cutmix 1.0 --reprob 0.25 --vit_dropout_rate 0 \
    --dist_eval \
    --data_path ${DATASET_DIR} \
    --num_workers 4 \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --dist_eval \
    --nb_classes 14 \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --seed 0 \
