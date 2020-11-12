# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:11:23 2020

@author: zll
"""


import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,
                 stride=1):
        '''
         convolution layer followed by batch normalization and active function.
        '''
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_channel)  # batch normalization
        self.ac = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ac(x)
        return x


class Double_Conv(nn.Module):
    '''
    Double convolution block with residual connection.
    '''
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,
                 stride=1):
        super(Double_Conv, self).__init__()
        # skip connection
        self.skip = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        # double convolution
        self.conv = nn.Sequential(
                Conv(in_channel, out_channel, kernel_size, padding, stride),
                Conv(out_channel, out_channel, kernel_size, padding, stride)
                )

    def forward(self, x):
        x1 = self.skip(x)
        x2 = self.conv(x)
        return x1 + x2  # residual connnection


class U_Net(nn.Module):
    '''
    U-Net.
    '''
    def __init__(self):
        super(U_Net, self).__init__()
        self.conv1 = Double_Conv(1, 32)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.conv2 = Double_Conv(32, 64)
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.conv3 = Double_Conv(64, 128)
        self.pooling3 = nn.MaxPool2d(2, 2)
        self.conv4 = Double_Conv(128, 256)

        self.conv_trans1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv5 = Double_Conv(256, 128)
        self.conv_trans2 = nn.ConvTranspose2d(128, 64)
        self.conv6 = Double_Conv(128, 64)
        self.conv_trans3 = nn.ConvTranspose2d(64, 32)
        self.conv7 = Double_Conv(64, 32)
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        # TODO
