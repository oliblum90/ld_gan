import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


def ones(bs):
    return Variable(torch.from_numpy(np.ones(bs))).cuda().float()
def zeros(bs):
    return Variable(torch.from_numpy(np.zeros(bs))).cuda().float()

    
class GanDis:
    
    def __init__(self, gen, dis, lr):
        
        self.criterion = nn.BCELoss()
        self.criterion.cuda()
        
        self.gen = gen
        self.dis = dis
        
        self.opt_dis = optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999))
        
    
    def train(self, X, Z):

        bs = X.size(0)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        self.dis.zero_grad()
        d = self.dis(X)
        errD_real = self.criterion(d, ones(bs))
        errD_real.backward()
        D_x = d.data.mean()

        # train with fake
        x = self.gen(Z)
        d = self.dis(x.detach())
        errD_fake = self.criterion(d, zeros(bs))
        errD_fake.backward()
        D_G_z1 = d.data.mean()
        errD = errD_real + errD_fake
        self.opt_dis.step()
        
        return errD