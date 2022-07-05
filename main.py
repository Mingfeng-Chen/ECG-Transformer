# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:30:01 2022

@author: ChenMingfeng
"""

import torch
import torch.nn as nn
from Transformer import ECGTransformer
from data import trainloader
import wandb

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    block_size = 16
    sample_size = 4
    
    model = ECGTransformer(block_size, sample_size, device=device).to(device)
    loss_func = nn.MSELoss().to(device)
    learning_rate = 1e-3
    num_epochs = 500
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    wandb.init(project="ECG_Transformer",
           config={ "learning_rate": learning_rate,
                    "batch_size": 32,
                    "total_run": num_epochs,
                    "network": model}
          )
    wandb.watch(model)
    
    for epoch in range(num_epochs):
        sum = 0
        for index, (x,y) in enumerate(trainloader):
            N = x.shape[0]
            x = x.reshape(N, 1, -1, 1).to(device)
            y = y.reshape(N, -1).to(device)
            out = model(x,y)
            loss = loss_func(out, y) 
            optim.zero_grad()
            loss.backward()
            optim.step()  
            a = out - y
            prd = torch.norm(a) / torch.norm(y)
            sum += prd.item()
        avg = sum / 112
        wandb.log({'prd':avg, 'epoch':epoch})
        if (epoch % 100 == 0 and epoch != 0) or (epoch == num_epochs - 1):
            torch.save(model.state_dict(), 'cr=50%_Transformer_checkpoint_'+str(epoch))
            print('save checkpoints on:', epoch, 'loss value is:', loss.item(), 'prd is:', avg)
            #print(optim.state_dict()['param_groups'][0]['lr'])
    wandb.finish()
        
