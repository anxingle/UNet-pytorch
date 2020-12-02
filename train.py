import argparse
import logging
import logging.config
from utils.load_conf import ConfigLoader
from pathlib import Path

logger_path = Path("./configs/logger.yaml")
conf = ConfigLoader(logger_path)
_logger = logging.getLogger(__name__)

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

base = Path(os.environ['raw_data_base']) if 'raw_data_base' in os.environ.keys() else Path('./data')
assert base is not None, "Please assign the raw_data_base(which store the training data) in system path "
dir_img = base / 'imgs'
dir_mask = base / 'masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(str(dir_img.resolve()), str(dir_mask.resolve()), img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    global_step = 0

    _logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # XXX: RMSprop算法实现-https://zh.d2l.ai/chapter_optimization/rmsprop.html
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # TODO: Dice loss/IOU loss?
    if net.n_classes > 1:
        # 使用普通交叉熵
        criterion = nn.CrossEntropyLoss()
    else:
        # 二分类使用二值交叉熵
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # imgs 形状应为 [BatchSize, Channel, Height, Width]
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                # 进度条右边显示内容
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % max((n_train // (10 * batch_size)), 1)== 0:
                    # 验证集评估模型
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)

                    if net.n_classes > 1:
                        _logger.info('Validation cross entropy: {}'.format(val_score))
                    else:
                        _logger.info('Validation Dice Coeff: {}'.format(val_score))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                _logger.info('Created checkpoint directory')
            except OSError:
                _logger.error('Failed to created checkpoint directory!')
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            _logger.info(f'Checkpoint {epoch + 1} saved !')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('--channels', type=int, default=3, help='image channels', dest='channels')
    parser.add_argument('--classes', type=int, default=1, help='mask nums', dest='classes')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f'Using device {device}')

    # 根据自己的数据进行调整
    # n_channels: 图片通道数，RGB彩色图片为3
    # n_classes: 每个像素的可能概率（候选）
    # n_classes=1: 前景与背景或两类object
    # n_classes=N: 类别N > 2
    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=True)
    _logger.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # 是否加载预训练模型
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        _logger.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # 快速卷积（更耗显存）
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        _logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
