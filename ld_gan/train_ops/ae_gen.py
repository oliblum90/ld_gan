import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

    
class AutoEncGen:
    
    def __init__(self, enc, gen, lr, freq=1):
        
        self.freq = freq
        
        self.criterion = nn.MSELoss()
        # self.criterion.size_average = False
        self.criterion.cuda()
        
        self.gen = gen
        self.enc = enc
        
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr)
        
    
    def train(self, X, Y, Z, batch_idxs, nn_idxs, sr_idxs, z_all):

        bs = X.size(0)

        ############################
        # (1) Update enc
        ###########################
        # train with real
        self.gen.zero_grad()
        z = self.enc(X)
        x = self.gen(z.detach())
        err = self.criterion(x, X)
        err.backward()
        mean_x = x.data.mean()
        self.opt_gen.step()
        
        return err.cpu().data.numpy()[0]
    
    
    
    
    