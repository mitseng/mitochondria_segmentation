# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:23:21 2020

@author: zll
"""

import numpy as np


def get_patch(img, x, y, aug):
    patch = np.copy(img[64*x:64*(x+1), 64*y:64*(y+1)])
    if aug == 0:
        ...