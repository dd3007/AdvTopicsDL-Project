# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from util.multi_label_loss import SoftTargetBinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_vit

from engine_med_finetune import train_one_epoch, evaluate_medical
from util.sampler import RASampler
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict

from util.dataloader_medical import CheXpert, ChestX_ray14
import torchvision.transforms as transforms

# NCCL is the protocol that should be used to communicate between GPUs
torch.distributed.init_process_group("nccl")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')


    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    # Dataset parameters

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument("--dataset", default='chestxray', type=str)

    parser.add_argument('--repeated-aug', action='store_true', default=False)

    parser.add_argument("--optimizer", default='adamw', type=str)
    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument("--checkpoint_type", default=None, type=str)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train = build_dataset_chest_xray(split='train', args=args)
    # dataset_val = build_dataset_chest_xray(split='val', args=args)
    # dataset_test = build_dataset_chest_xray(split='test', args=args)

    # dataset_name = 'chexpert'
    dataset_name = 'chestxray_nih'

    mean_dict = { 'chexpert': [0.485, 0.456, 0.406], 'chestxray_nih': [0.5056, 0.5056, 0.5056] }
    std_dict = { 'chexpert': [0.229, 0.224, 0.225], 'chestxray_nih': [0.252, 0.252, 0.252] }

    # args variables not used here
    random_resize_range = None
    mask_strategy = None

    dataset_mean = mean_dict[dataset_name]
    dataset_std = std_dict[dataset_name]
        
    if random_resize_range:
        if mask_strategy in ['heatmap_weighted', 'heatmap_inverse_weighted']:
            resize_ratio_min, resize_ratio_max = random_resize_range
            print(resize_ratio_min, resize_ratio_max)
            # transform_train = custom_train_transform(size=args['input_size'],
                                                        # scale=(resize_ratio_min, resize_ratio_max),
                                                        # mean=dataset_mean, std=dataset_std)
        else:
            resize_ratio_min, resize_ratio_max = random_resize_range
            print(resize_ratio_min, resize_ratio_max)
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(resize_ratio_min, resize_ratio_max),
                                                interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)])
    else:
        print('Using Directly-Resize Mode. (no RandomResizedCrop)')
        transform_train = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)]
        )

    heatmap_path = None
    if mask_strategy in ['heatmap_weighted', 'heatmap_inverse_weighted']:
        heatmap_path = 'nih_bbox_heatmap.png'

    if dataset_name == 'chexpert':
        dataset_train = CheXpert(csv_path="/mnt/home/mpaez/ceph/CheXpert-v1.0-small/train.csv", image_root_path='/mnt/home/mpaez/ceph/CheXpert-v1.0-small', use_upsampling=False,
                            use_frontal=True, mode='train', class_index=-1, transform=transform_train,
                            heatmap_path=heatmap_path, pretraining=False)
        dataset_val = CheXpert(csv_path="/mnt/home/mpaez/ceph/CheXpert-v1.0-small/valid.csv", image_root_path='/mnt/home/mpaez/ceph/CheXpert-v1.0-small', use_upsampling=False,
                            use_frontal=True, mode='valid', class_index=-1, transform=transform_train,
                            heatmap_path=heatmap_path, pretraining=False)
        #dataset_test = CheXpert(csv_path="data/chexpert/test.csv", image_root_path='data/chexpert/', use_upsampling=False,
        #            use_frontal=True, mode='test', class_index=-1, transform=transform_train,
        #            heatmap_path=heatmap_path, pretraining=False)
    elif dataset_name == 'chestxray_nih':
        dataset_train = ChestX_ray14('/mnt/home/mpaez/ceph/chestxray/images', '/mnt/home/mpaez/ceph/chestxray/train_official.txt', augment=transform_train, num_class=14,
                                heatmap_path=heatmap_path, pretraining=False)
        dataset_val = ChestX_ray14('/mnt/home/mpaez/ceph/chestxray/images', '/mnt/home/mpaez/ceph/chestxray/val_official.txt', augment=transform_train, num_class=14,
                                heatmap_path=heatmap_path, pretraining=False)
        dataset_test = ChestX_ray14('/mnt/home/mpaez/ceph/chestxray/images', '/mnt/home/mpaez/ceph/chestxray/test_official.txt', augment=transform_train, num_class=14,
                                heatmap_path=heatmap_path, pretraining=False)
    else:
        raise NotImplementedError

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            # sampler_val = torch.utils.data.DistributedSampler(
            #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    if 'vit' in args.model:
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_rate=args.vit_dropout_rate,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    elif 'densenet' in args.model or 'resnet' in args.model:
        model = models.__dict__[args.model](num_classes=args.nb_classes)
    else:
        raise NotImplementedError


    if args.finetune and not args.eval:
        if 'vit' in args.model:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in checkpoint_model.keys():
                if k in state_dict:
                    if checkpoint_model[k].shape == state_dict[k].shape:
                        state_dict[k] = checkpoint_model[k]
                        print(f"Loaded Index: {k} from Saved Weights")
                    else:
                        print(f"Shape of {k} doesn't match with {state_dict[k]}")
                else:
                    print(f"{k} not found in Init Model")



            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)


            # if args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)
        elif 'densenet' in args.model or 'resnet' in args.model:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'state_dict' in checkpoint.keys():
                checkpoint_model = checkpoint['state_dict']
            elif 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            if args.checkpoint_type == 'smp_encoder':
                state_dict = checkpoint_model

                new_state_dict = OrderedDict()

                for key, value in state_dict.items():
                    if 'model.encoder.' in key:
                        new_key = key.replace('model.encoder.', '')
                        new_state_dict[new_key] = value
                checkpoint_model = new_state_dict
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if 'vit' in args.model:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    # elif args.optimizer == 'fusedlamb':
    #     optimizer = FusedAdam(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.dataset == 'chestxray':
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetBinaryCrossEntropy()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif args.dataset == 'chexpert':
        criterion = losses.CrossEntropyLoss()
    else:
        raise NotImplementedError
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # if
    # criterion = torch.nn.BCEWithLogitsLoss()


    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate_chestxray(data_loader_test, model, device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.eval_interval == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

            test_stats = evaluate_chestxray(data_loader_test, model, device, args)
            print(f"Average AUC on the test set images: {test_stats['auc_avg']:.4f}")
            max_auc = max(max_auc, test_stats['auc_avg'])
            print(f'Max Average AUC: {max_auc:.4f}', {max_auc})

            if log_writer is not None:
                log_writer.add_scalar('perf/auc_avg', test_stats['auc_avg'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)