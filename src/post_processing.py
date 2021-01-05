# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:44:40 2021

@author: zll
"""

import os
import cv2 as cv


src_path = '../pred_6/'
dist_path = '../rst/'

src_files = sorted(os.listdir(src_path))
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
for f in src_files:
    img = cv.imread(src_path + f, cv.IMREAD_GRAYSCALE)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=7)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
    cv.imwrite(dist_path + f, img)
