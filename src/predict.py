# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:12:03 2020

@author: zll
"""


import cv2 as cv
import torch
import numpy as np
import os
from model import U_Net


def seg_img(img_file, model, device):
    '''returns prediction of numpy 2darray of uint8 in main memory.'''
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
    # image size is 768 * 1024, no need padding
    img = img / 255
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))
    inputs = torch.from_numpy(img)
    inputs = inputs.float()
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputs = outputs.to('cpu')
    outputs = outputs[0, 1, :, :]
    outputs = np.uint8(outputs >= 0.5)
    outputs *= 255
    return outputs


if __name__ == '__main__':
    model_path = './param_32/param480.pkl'
    out_path = '../pred_5/'
    img_path = '../../mito_imgs/test/images/'
    device = 'cuda:0'
    img_files = sorted(os.listdir(img_path))
    unet = U_Net()
    unet.eval()
    unet.load_state_dict(torch.load(model_path,
                                    map_location=torch.device(device)))
    unet = unet.to(device)
    for f in img_files:
        pred = seg_img(img_path + f, unet, device)
        cv.imwrite(out_path + f, pred)
