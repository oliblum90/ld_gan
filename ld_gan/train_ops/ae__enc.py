import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

    
class AutoEncEnc:
    
    def __init__(self, enc, gen, lr):
        
        self.criterion = nn.MSELoss()
        # self.criterion.size_average = False
        self.criterion.cuda()
        
        self.gen = gen
        self.enc = enc
        
        self.opt_enc = optim.Adam(self.enc.parameters(), lr=lr)
        
    
    def train(self, X, Y, Z, Z_bar):

        bs = X.size(0)

        ############################
        # (1) Update enc
        ###########################
        # train with real
        self.enc.zero_grad()
        z = self.enc(X)
        x = self.gen(z.detach())
        err = self.criterion(x, X)
        err.backward()
        mean_x = x.data.mean()
        self.opt_enc.step()
        
        return err.cpu().data.numpy()[0]
    
    
    
    
    