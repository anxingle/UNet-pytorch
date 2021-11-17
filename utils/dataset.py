from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
import cv2
from torch.utils.data import Dataset
import logging
_logger = logging.getLogger(__name__)

from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        _logger.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, gray=False):
        h, w, c = pil_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        #  scale image
        pil_img = cv2.resize(pil_img, (newW, newH))
        # convert to 1 channel
        if gray:
            pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2GRAY)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(self.masks_dir + '/' + idx + self.mask_suffix + '.*')
        # img_file = glob(self.imgs_dir + '/' + idx + '.*')
        mask_file = os.path.join(self.masks_dir, '%s' % idx + self.mask_suffix + '.bmp')
        img_file = os.path.join(self.imgs_dir, '%s' % idx + '.bmp')

        # XXX: 这里mask_file 应该是路径名
        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])
        mask = cv2.imread(mask_file)
        img = cv2.imread(img_file)

        assert img is not None, "img 为空，请检查图片是否在该路径下：%s" % img_file
        assert mask is not None, "mask 为空，请检查图片是否在该路径下：%s" % mask_file

        assert img.shape == mask.shape, \
            f'Image 和 mask {idx} 应该有相同的大小, 但这里img是： {img.shape}， mask是： {mask.shape}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale, gray=True)

        assert img.shape[1] == mask.shape[1], 'img and mask must have same height   %s' % idx
        assert img.shape[2] == mask.shape[2], 'img and mask must have same width    %s' % idx

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

if __name__ == '__main__':
    data = BasicDataset('/Volumes/data/workspace/cv/Pytorch-UNet/data/imgs/', '/Volumes/data/workspace/cv/Pytorch-UNet/data/masks/')
    for n, i in enumerate(data):
        print(n)




