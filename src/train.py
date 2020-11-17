# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:23:58 2020

@author: zll
"""


import torch
from Mito_Dataset import Mito_Dataset
from torch.utils.data import DataLoader
from model import U_Net
from time import time


# ******** Hyper Parameters ********
# epoches till stop
EPOCHES = 1000
# if there is pretrained parameters
PRETRAIN = False
# epoches trained
pre_epoch = 0
# pretrained model parameter
pretrained = ''
# batch size
batch_size = 16
# **********************************


dataset = Mito_Dataset()
data_loader = DataLoader(dataset, batch_size=batch_size)
print('Data loaded.')

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# new and init model
model = U_Net()
if PRETRAIN:
    model.load_state_dict(torch.load(pretrained))
model.to(device)  # copy model to GPU

# loss function
criterion = torch.nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


# training

# training
for epoch in range(pre_epoch + 1, pre_epoch + 1 + EPOCHES):
    start_time = time()
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        # get input
        inputs, lables = data
        inputs = torch.tensor(inputs, dtype=torch.float32)
        lables = torch.tensor(lables, dtype=torch.long)
        lables = lables.squeeze(1)
        # copy data to GPU
        inputs, lables = inputs.to(device), lables.to(device)
        # set gradiant 0
        optimizer.zero_grad()
        # forwarding
        
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, lables)
        # backwarding
        loss.backward()
        # optimizing
        optimizer.step()
        # print states info
        running_loss += loss.item() * batch_size
    # print epoch loss and time
    print('[%d, loss: %.6f]' % (epoch, running_loss / dataset.len))
    print((time() - start_time) // 60, 'minutes per epoch.')
    # save model every epoch every 10 epoch
    if epoch % 10 == 0:
        torch.save(model.state_dict(), './param_16/param'+str(epoch)+'.pkl')
