EXP_NAME=distill_finetuned_tiny_model
GPUS=4
SAVE_DIR="./work_dirs/${EXP_NAME}_e1/"

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --use_env main_finetune_chestxray.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 32 \
    --finetune "tiny_mae_pretrained.pth" \
    --epochs 70 \
    --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 \
    --model vit_tiny_patch16 \
    --warmup_epochs 5 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --num_workers 4 \
    --nb_classes 14 \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1'
    # --data_path ${DATASET_DIR} \
    # --train_list ${TRAIN_LIST} \
    # --val_list ${VAL_LIST} \
    # --test_list ${TEST_LIST} \