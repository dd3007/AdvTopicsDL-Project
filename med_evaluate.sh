EXP_NAME=evaluation_chestxray14_small_tiny_100epochs_4gpus
SAVE_DIR="./work_dirs/${EXP_NAME}_e1/"
GPUS=4

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --use_env main_med_finetune.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 32 \
    --model vit_tiny_patch16 \
    --dataset chestxray14 \
    --nb_classes 14 \
    --epochs 100 \
    --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 \
    --warmup_epochs 5 \
    --drop_path 0.2 \
    --mixup 0 --cutmix 0 \
    --vit_dropout_rate 0 \
    --num_workers 4 \
    --eval_interval 10 \
    --resume "finetuned_small_tiny_chestxray14_100epochs.pth" \
    --eval