# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:41:46 2020

@author: zll
"""


import cv2 as cv
import numpy as np
import os
from metrics import metrics


def get_imgarr(img_files):
    imgarr = []
    for f in img_files:
        imgarr.append(cv.imread(f))
    return np.uint8(imgarr)


predict_path = '../pred_5/'
lable_path = '../../mito_imgs/annotations/'

predict_files = os.listdir(predict_path)
predict_files = [predict_path + f for f in predict_files]
lable_files = os.listdir(lable_path)
lable_files = [lable_files + f for f in lable_files]

pred_img_arr = get_imgarr(predict_files)
lable_img_arr = get_imgarr(lable_files)

met = metrics(pred_img_arr, lable_img_arr)
print(met)
