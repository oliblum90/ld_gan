import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


def ones(bs):
    return Variable(torch.from_numpy(np.ones(bs))).cuda().float()
def zeros(bs):
    return Variable(torch.from_numpy(np.zeros(bs))).cuda().float()

    
class GanEnc:
    
    def __init__(self, enc, gen, dis, lr, freq=1):
        
        self.freq = freq
        
        self.criterion = nn.BCELoss()
        self.criterion.cuda()
        
        self.gen = gen
        self.dis = dis
        self.enc = enc
        
        self.opt_enc = optim.Adam(self.enc.parameters(), lr=lr, betas=(0.5, 0.999))
        
    
    def train(self, X, Y, Z, Z_bar):

        bs = X.size(0)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.enc.zero_grad()
        z = self.enc(X)
        x = self.gen(z)
        d = self.dis(x)
        errE = self.criterion(d, ones(bs))
        errE.backward()
        D_G_z2 = d.data.mean()
        self.opt_enc.step()
        
        return errE.cpu().data.numpy()[0]