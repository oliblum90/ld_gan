import torch.nn as nn
from init_weights import init_weights
import numpy as np

    
class Gen(nn.Module):
    
    def __init__(self, 
                 latent_size     = 256,
                 ipt_size        = 64,
                 complexity      = 64,
                 n_col           = 3,
                 w_norm          = False,
                 batchnorm  = True):
        
        super(Gen, self).__init__()
        
        if w_norm:
            from WeightNormalizedConv import WeightNormalizedConvTranspose2d as CT
        else:
            from torch.nn import ConvTranspose2d as CT
        
        self.n_blocks = int(np.log2(ipt_size) - 1)
        self.main = nn.Sequential()
        self.latent_size = latent_size
        
        # BLOCK 0
        c_out = complexity * 2**(self.n_blocks - 2)
        self.main.add_module('b00', CT(latent_size, c_out, 4, 1, 0, bias=False))
        if batchnorm:
            self.main.add_module('b01', nn.BatchNorm2d(c_out))
        self.main.add_module('b02', nn.ReLU(True))

        # BLOCKS 1 - N-1
        for b in reversed(range(1, self.n_blocks - 1)):
            c_in  = complexity * 2**(b)
            c_out = complexity * 2**(b-1)
            n = 'b' + str(b)
            self.main.add_module(n+'0', CT(c_in, c_out, 4, 2, 1, bias=False))
            if batchnorm:
                self.main.add_module(n+'1', nn.BatchNorm2d(c_out))
            self.main.add_module(n+'2', nn.ReLU(True))
        
        # BLOCK N: 4 --> 1
        n = 'b' + str(self.n_blocks-1)
        self.main.add_module(n+"0", CT(complexity, n_col, 4, 2, 1, bias=False))
        self.main.add_module(n+"1", nn.Tanh())
       
    
    def forward(self, x):
                
        x = self.main(x)
        
        return x
    
    