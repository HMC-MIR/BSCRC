import torch
import torch.nn as nn
import numpy as np

from argparse import ArgumentParser
import os
import math
import sys
import json
import time
import datetime
from pathlib import Path

from utils import *
from models import MaskedAutoencoderViT
from functools import partial
import timm.optim.optim_factory as optim_factory


def train(args):

    # ##################################################
    #              Configuration Setup                 #
    # ##################################################

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ##################################################
    #               Create DataLoader                  #
    # ##################################################

    train_dataset, val_dataset, _, _, _ = create_dataset(args.data_path, pretrain=True)
    print(f"Loaded data of size {len(train_dataset)} and {len(val_dataset)}")
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=False)

    # ##################################################
    #                 Define Model                     #
    # ##################################################

    model = MaskedAutoencoderViT(
        img_size=64, patch_size=8, in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # ##################################################
    #                   Optimizer                      #
    # ##################################################

    eff_batch_size = args.batch_size * torch.cuda.device_count()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # ##################################################
    #                   Training                       #
    # ##################################################

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_data_loader, optimizer, loss_scaler, epoch, args)
        val_stats = evaluate(val_data_loader, model, args)

        save_model(args, epoch, model, loss_scaler, optimizer)

        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}
            f.write(json.dumps(log_stats) + "\n")

    # ##################################################
    #                 Save Configs                     #
    # ##################################################

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    with open(os.path.join(args.output_dir, "config.txt"), mode="a", encoding="utf-8") as f:
        f.write(f'Model = {str(model.module)}\n\n')
        f.write(f'{args}'.replace(', ', '\n') + '\n\n')
        f.write(f'Training time: {total_time}\n\n')


def train_one_epoch(model, data_loader, optimizer, loss_scaler, epoch, args):

    model.train(True)
    optimizer.zero_grad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, 10, f'Epoch {epoch}')):

        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device)

        # with torch.cuda.amp.autocast():
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        loss = loss.sum()

        if not math.isfinite(loss):
            print(f"Loss is {loss}, stopping training")
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger, "\n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, args):

    model.eval()  # switch to evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_logger = MetricLogger(delimiter="  ")

    for (samples, _) in metric_logger.log_every(data_loader, 10, 'Validation '):

        samples = samples.to(device)

        # with torch.cuda.amp.autocast():
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        metric_logger.update(loss=loss.sum())

    print('* Validation loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():

    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, metavar='PATH', help='path to pkl file')
    parser.add_argument('--output_dir', required=True, type=str,  metavar='PATH', help='path where to save')
    parser.add_argument('--mask_ratio', required=True, type=float, metavar='%', help='percntage of removed patches')
    parser.add_argument('--epochs', required=True, type=int, metavar='N', help='total epochs')
    parser.add_argument('--warmup_epochs', required=True, type=int, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--batch_size', required=True, type=int, metavar='N', help='samples per batch')

    # optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='absolute lr')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--clip_grad', type=int, default=None)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()
