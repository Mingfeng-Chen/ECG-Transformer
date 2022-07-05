# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:38:06 2022

@author: ChenMingfeng
"""

import torch
import torch.nn as nn
from Transformer import ECGTransformer
from data import testloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

block_size = 16
sample_size = 8

model = ECGTransformer(block_size, sample_size, device=device).to(device)
loss_func = nn.MSELoss().to(device)

model.eval()

for i in range(1, 11):
    model.load_state_dict(torch.load('cr=50%_Transformer_checkpoint_'+str(i*100)))
    sum = 0
    #file = open(str(1100)+"test 3rd.txt",'w')

    for index, (x,y) in enumerate(testloader):
        N = x.shape[0]
        x = x.reshape(N, 1, -1, 1).to(device)
        y = y.reshape(N, -1).to(device)
        out = model(x, y)
        loss = loss_func(out, y)
        a = out - y
        prd = torch.norm(a) / torch.norm(y)
        sum += prd.item()
        #print(index, loss.item(), prd.item())
        #file.write(str(prd.item())+"\n")
        
    #file.close()
    avg = sum / 112
    print(avg)
    
    