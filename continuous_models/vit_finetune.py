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
from models import VisionTransformer
from functools import partial
from timm.models.layers import trunc_normal_
from timm.utils import accuracy


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

    train_dataset, val_dataset, _, _, _ = create_dataset(args.data_path)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=False)

    # ##################################################
    #                 Define Model                     #
    # ##################################################

    if args.probe:
        model = VisionTransformer(
            img_size=64, patch_size=8, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=args.nb_classes,
            global_pool=args.global_pool)
    else:
        model = VisionTransformer(
            img_size=64, patch_size=8, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=args.nb_classes,
            drop_path_rate=args.drop_path, global_pool=args.global_pool)

    # ##################################################
    #           Finetune Pretrained Model              #
    # ##################################################

    if args.finetune:
        print(f"Load pre-trained from: {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # remove head.weight and head.bias keys from pretrained if exists and don't match shape
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # load pretrained model, ignore non-matching keys
        msg = model.load_state_dict(checkpoint_model, strict=False) # load pre-trained model
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        # manually initialize fc layer
        if args.probe:
            trunc_normal_(model.head.weight, std=0.01)
        else:
            trunc_normal_(model.head.weight, std=2e-5)

    if args.probe:
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # ##################################################
    #                   Optimizer                      #
    # ##################################################

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_batch_size = args.batch_size * torch.cuda.device_count()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.probe:
        print("Training linear probe")
        optimizer = LARS(model.module.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Finetuning model")
        param_groups = param_groups_lrd(model.module, args.weight_decay,
            no_weight_decay_list=model.module.no_weight_decay(), layer_decay=args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    # load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # ##################################################
    #                   Training                       #
    # ##################################################

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):

        train_stats = train_one_epoch(model, criterion, train_data_loader, optimizer, epoch, loss_scaler, args)
        val_stats = evaluate(val_data_loader, model)
        print(f"Accuracy on {len(val_dataset)} val images: {val_stats['acc1']:.1f}%")

        save_model(args, epoch, model, loss_scaler, optimizer)

        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch, 'n_parameters': n_parameters}
            f.write(json.dumps(log_stats) + "\n")

    # ##################################################
    #                 Save Configs                     #
    # ##################################################

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    with open(os.path.join(args.output_dir, "config.txt"), mode="a", encoding="utf-8") as f:
        f.write(f'Model = {str(model.module)}\n\n')
        f.write('Number of params (M): %.2f' % (n_parameters / 1.e6) + '\n\n')
        f.write(f'{args}'.replace(', ', '\n') + '\n\n')
        f.write(f'Training time: {total_time}\n\n')


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, loss_scaler, args):

    model.train(True)
    optimizer.zero_grad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, f'Epoch {epoch}')):

        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets = samples.to(device), targets.to(device)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

    print("Averaged stats:", metric_logger, "/n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model):

    model.eval()  # switch to evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")

    for batch in metric_logger.log_every(data_loader, 10, 'Validation '):

        images, target = batch[0], batch[-1]
        images, target = images.to(device), target.to(device)

        # with torch.cuda.amp.autocast():
        output = model(images) # compute output
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():

    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, metavar='PATH', help='path to pkl file')
    parser.add_argument('--output_dir', type=str,  metavar='PATH', help='path where to save')
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--nb_classes', type=int, help='number of the classification types')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='total epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='samples per batch')
    parser.add_argument('--probe', action='store_true', help='fit linear probe')
    parser.set_defaults(probe=False)

    # optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    # parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')

    # finetune
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # NO augmentation params (color_jitter, aa, smoothing)
    # NO Random Erase params (reprob, remode, recount, resplit)
    # NO mixup params (mixup, cutmix, cutmix_minmax, mixup_prob, mixup_switch_prob, mixup_mode)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
