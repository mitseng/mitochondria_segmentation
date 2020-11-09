# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:23:21 2020

@author: zll
"""

import numpy as np


def get_patch(img, x, y, aug):
    '''
    return augmented image patch.
    img: numpy array of uint8,
    x: patch index, [0, 11]
    y: patch index, [0, 15]
    aug: augment method
    '''
    patch = np.copy(img[64*x:64*(x+1), 64*y:64*(y+1)])
    if aug < 4:  # rotate 0, 90, 180, 270 degrees
        patch = np.rot90(patch, aug)
    else:  # flip vertically, horizongtally
        patch = np.flip(patch, aug - 4)
    return patch
