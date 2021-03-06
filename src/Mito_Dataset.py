# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:15:22 2020

@author: zll
"""

import os
import cv2 as cv
from torch.utils import data
from patch_extraction import get_patch


class Mito_Dataset(data.Dataset):
    def __init__(self):
        self.train_imgs = 126
        self.patch_size = (64, 64)
        self.len = (768 // 64) * (1024 // 64) * 6 * 126  # 145152
        self.train_dir = '../../mito_imgs/train/images/'
        self.test_dir = '../../mito_imgs/train/annotations/'
        self.train_files = sorted(os.listdir(self.train_dir))
        self.test_files = sorted(os.listdir(self.test_dir))
        self.train_files.sort()
        self.test_files.sort()

        self.img_idx = -1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index // 1152 != self.img_idx:
            self.img_idx = index // 1152
            self.train_img = cv.imread(self.train_dir + self.train_files[self.img_idx],
                                       cv.IMREAD_GRAYSCALE)
            self.test_img = cv.imread(self.test_dir + self.test_files[self.img_idx],
                                      cv.IMREAD_GRAYSCALE)
        patch_idx = index % 1152 // 6
        patch_x, patch_y = patch_idx // 16, patch_idx % 16
        augment = index % 1152 % 6
        inputs = get_patch(self.train_img, patch_x, patch_y, augment).reshape((1, 64, 64))
        targets = get_patch(self.test_img, patch_x, patch_y, augment).reshape((1, 64, 64))
        # inputs = np.float64(inputs)
        inputs = inputs / 255
        targets = targets / 255
        return (inputs.copy(), targets.copy())
