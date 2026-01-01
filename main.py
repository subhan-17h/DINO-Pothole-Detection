# Copyright (c) 2022 IDEA. All Rights Reserved.
#type: ignore
# ------------------------------------------------------------------------
"""python main.py `
   --config_file config/DINO/DINO_4scale.py `
   --output_dir outputs/pothole_finetune `
   --coco_path datasets/pothole `
   --pretrain_model_path checkpoints/checkpoint0011_4scale.pth `
   --options epochs=10 lr=1e-5 batch_size=1 lr_drop=8 `
   --num_workers 2 `
   --device cuda `
   --amp"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test

# Enhanced logging and progress tracking imports
from tqdm import tqdm
import logging



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def print_section_header(title, width=80):
    """Print a clean section header"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")

def print_config_summary(args):
    """Print a clean summary of configuration"""
    print_section_header("CONFIGURATION SUMMARY")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🎯 Config File: {args.config_file}")
    print(f"📊 Dataset: {args.dataset_file}")
    print(f"🗂️  COCO Path: {args.coco_path}")
    print(f"🚀 Device: {args.device}")
    print(f"🏃‍♂️ Batch Size: {args.batch_size}")
    print(f"📈 Learning Rate: {args.lr}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"⚡ AMP Enabled: {args.amp}")
    print(f"💾 Pretrained Model: {args.pretrain_model_path}")
    print_section_header("TRAINING SETUP")

def format_time(seconds):
    """Format time in human readable format"""
    return str(datetime.timedelta(seconds=int(seconds)))

def print_epoch_progress(epoch, total_epochs, stage="", metrics=None):
    """Print clean epoch progress"""
    progress_bar = "█" * int((epoch + 1) / total_epochs * 30) + "░" * int((total_epochs - epoch - 1) / total_epochs * 30)
    print(f"\n{'🔄 ' if stage else ''}Epoch {epoch + 1}/{total_epochs} [{progress_bar}] {stage}")
    if metrics:
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    print_section_header("DINO TRAINING SCRIPT")
    print_section_header("INITIALIZATION")

    utils.init_distributed_mode(args)

    # Clean config loading output
    print(f"📋 Loading config from: {args.config_file}")
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        print(f"⚙️  Merged options: {args.options}")

    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger with cleaner output
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"💾 Full config saved to: {save_json_path}")

    # Print configuration summary
    print_config_summary(args)

    # Log essential info to file
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print_section_header("MODEL SETUP")

    device = torch.device(args.device)
    print(f"🎯 Using device: {device}")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"🌱 Random seed set to: {seed}")

    # build model
    print("🏗️  Building model...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    print("✅ Model built and moved to device")

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
        print(f"📊 EMA enabled with decay: {args.ema_decay}")
    else:
        ema_m = None
        print("📊 EMA disabled")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
        print(f"🌐 Distributed training enabled (GPU: {args.gpu})")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total trainable parameters: {n_parameters:,}")
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    print(f"⚙️  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    

    print_section_header("DATASET SETUP")

    print("📊 Building datasets...")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print(f"✅ Training samples: {len(dataset_train):,}")
    print(f"✅ Validation samples: {len(dataset_val):,}")

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        print("🌐 Using distributed samplers")
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        print("🔧 Using local samplers")

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    print(f"🔄 Creating data loaders (workers: {args.num_workers})...")
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    print(f"✅ Data loaders ready (batches per epoch: {len(data_loader_train)})")

    print_section_header("SCHEDULER SETUP")
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
        print("📈 Using OneCycleLR scheduler")
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
        print(f"📈 Using MultiStepLR scheduler (milestones: {args.lr_drop_list})")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        print(f"📈 Using StepLR scheduler (drop every {args.lr_drop} epochs)")


    print_section_header("CHECKPOINT & RESUME SETUP")

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)

    # Check for existing checkpoint
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        print(f"🔄 Found existing checkpoint: {args.resume}")

    # Resume training if checkpoint specified
    if args.resume:
        print_section_header("RESUMING TRAINING")
        if args.resume.startswith('https'):
            print("🌐 Downloading checkpoint from URL...")
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print(f"📂 Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        print("✅ Model weights loaded")

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
                print("✅ EMA weights loaded")
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)
                print("⚠️  EMA weights not found, creating new EMA model")

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resuming from epoch {args.start_epoch}")

    # Load pretrained model if specified
    if (not args.resume) and args.pretrain_model_path:
        print_section_header("LOADING PRETRAINED MODEL")
        print(f"🎯 Loading pretrained model from: {args.pretrain_model_path}")
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
        print(f"✅ Pretrained model loaded (missing keys: {len(_load_output.missing_keys)}, unexpected: {len(_load_output.unexpected_keys)})")

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
                print("✅ EMA weights loaded from pretrained model")
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)
                print("📊 Created new EMA model from pretrained weights")        


    # Evaluation mode
    if args.eval:
        print_section_header("EVALUATION MODE")
        os.environ['EVAL_FLAG'] = 'TRUE'
        print("🔍 Running evaluation...")

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)

        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            print(f"💾 Evaluation results saved to: {output_dir / 'eval.pth'}")

        # Print evaluation results
        print_section_header("EVALUATION RESULTS")
        for key, value in test_stats.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                print(f"📊 {key}: {value[0]:.4f}")
            else:
                print(f"📊 {key}: {value}")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # Training mode
    print_section_header("TRAINING MODE")
    print("🚀 Starting training...")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(args.start_epoch, args.epochs),
                      desc="Training Progress",
                      unit="epoch",
                      dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # Update progress bar description with current epoch
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        if args.distributed:
            sampler_train.set_epoch(epoch)

        # Training phase
        print_section_header(f"TRAINING - EPOCH {epoch + 1}")
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)

        # Update progress bar with training loss
        if 'loss' in train_stats:
            epoch_pbar.set_postfix({'train_loss': f"{train_stats['loss']:.4f}"})

        # Learning rate scheduler step
        if not args.onecyclelr:
            lr_scheduler.step()

        # Save checkpoint
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

            if len(checkpoint_paths) > 1:
                print(f"💾 Checkpoint saved: {checkpoint_paths[-1]}")
                
        # Evaluation phase
        print_section_header(f"EVALUATION - EPOCH {epoch + 1}")
        print("🔍 Running validation...")
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )

        # Extract and display mAP
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)

        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print_section_header(f"EPOCH {epoch + 1} RESULTS")
        print(f"📈 mAP (Regular): {map_regular:.4f} {'🏆 NEW BEST!' if _isbest else ''}")
        print(f"⏱️  Epoch Time: {format_time(epoch_time)}")

        # Save best model
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            print(f"🏆 New best model saved!")

        # EMA evaluation
        if args.use_ema:
            print("🔍 Running EMA validation...")
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest_ema = best_map_holder.update(map_ema, epoch, is_ema=True)
            print(f"📈 mAP (EMA): {map_ema:.4f} {'🏆 NEW BEST!' if _isbest_ema else ''}")

            if _isbest_ema:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
                print(f"🏆 New best EMA model saved!")

        # Update progress bar with latest mAP
        epoch_pbar.set_postfix({
            'train_loss': f"{train_stats.get('loss', 0):.4f}",
            'mAP': f"{map_regular:.4f}"
        })

        # Convert tensors to Python scalars for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):  # Tensor
                return obj.item()
            elif isinstance(obj, list):
                return [convert_to_serializable(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            else:
                return obj

        # Prepare logging statistics
        train_stats_serializable = convert_to_serializable(train_stats)
        test_stats_serializable = convert_to_serializable(test_stats)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats_serializable.items()},
            **{f'test_{k}': v for k, v in test_stats_serializable.items()},
        }

        if args.use_ema:
            ema_test_stats_serializable = convert_to_serializable(ema_test_stats)
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats_serializable.items()})

        log_stats.update(best_map_holder.summary())
        log_stats.update({
            'epoch': epoch,
            'n_parameters': n_parameters,
            'epoch_time': format_time(epoch_time)
        })

        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        # Save logs
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # Save evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    # Training completion
    epoch_pbar.close()
    total_time = time.time() - start_time
    total_time_str = format_time(total_time)

    print_section_header("TRAINING COMPLETED")
    print(f"🎉 Training finished successfully!")
    print(f"⏱️  Total Training Time: {total_time_str}")
    print(f"📁 Output Directory: {args.output_dir}")

    # Print best metrics summary
    best_summary = best_map_holder.summary()
    if best_summary:
        print_section_header("BEST METRICS")
        for key, value in best_summary.items():
            if isinstance(value, float):
                print(f"🏆 {key}: {value:.4f}")
            else:
                print(f"🏆 {key}: {value}")

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        print_section_header("CLEANUP")
        for filename in copyfilelist:
            print(f"🗑️  Removing: {filename}")
            remove(filename)

    print_section_header("SESSION COMPLETE")
    print("✅ All tasks completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
