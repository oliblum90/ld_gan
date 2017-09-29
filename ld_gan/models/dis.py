import torch.nn as nn
from init_weights import init_weights
import numpy as np
import torch.nn.functional as F
import torch
from WeightNormalizedConv import WeightNormalizedConv2d as WNC
from WeightNormalizedConv import WeightNormalizedConvTranspose2d as WNCT
    
    
class Dis(nn.Module):
    
    def __init__(self, 
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3):
        
        super(Dis, self).__init__()
        
        self.n_blocks = int(np.log2(ipt_size) - 1)
        self.main = nn.Sequential()
        
        # BLOCK 0
        self.main.add_module('b00', nn.Conv2d(n_col, complexity, 4, 2, 1, bias=False))
        self.main.add_module('b01', nn.LeakyReLU(0.2, inplace=True))

        # BLOCKS 1 - N-1
        for b in range(1, self.n_blocks - 1):
            c_in  = complexity * 2**(b-1)
            c_out = complexity * 2**(b)
            n = 'b' + str(b)
            self.main.add_module(n+'0', nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False))
            self.main.add_module(n+'1', nn.BatchNorm2d(c_out))
            self.main.add_module(n+'2', nn.LeakyReLU(0.2, inplace=True))
        
        # BLOCK N: 4 --> 1
        n = 'b' + str(self.n_blocks-1)
        self.main.add_module(n+"0", nn.Conv2d(c_out, 1, 4, 1, 0, bias=False))
        self.main.add_module(n+"1", nn.Sigmoid())
       
    
    def forward(self, x):
        x = self.main(x)
        return x
    
    
    
    
class InfoDis(nn.Module):
    
    def __init__(self, 
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3,
                 n_features = 256):
        
        super(InfoDis, self).__init__()
        
        self.n_blocks = int(np.log2(ipt_size) - 1)
        self.main = nn.Sequential()
        
        # BLOCK 0
        self.main.add_module('b00', nn.Conv2d(n_col, complexity, 4, 2, 1, bias=False))
        self.main.add_module('b01', nn.LeakyReLU(0.2, inplace=True))

        # BLOCKS 1 - N-1
        for b in range(1, self.n_blocks - 1):
            c_in  = complexity * 2**(b-1)
            c_out = complexity * 2**(b)
            n = 'b' + str(b)
            self.main.add_module(n+'0', nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False))
            self.main.add_module(n+'1', nn.BatchNorm2d(c_out))
            self.main.add_module(n+'2', nn.LeakyReLU(0.2, inplace=True))
        
        # BLOCK N: 4 --> 1
        n = 'b' + str(self.n_blocks-1)
        self.main.add_module(n+"0", nn.Conv2d(c_out, n_features, 4, 1, 0, bias=False))
        
        self.fc0 = nn.Linear(2*n_features, n_features)
        self.bn0 = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, 1)

       
    
    def forward(self, x, z):
        x = self.main(x)
        x = torch.cat((x,z), 1)
        x = x.view(-1, x.size(1))
        x = self.fc0(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.bn0(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x
    
    