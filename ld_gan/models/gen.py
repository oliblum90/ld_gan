import torch.nn as nn
from torch.nn import ConvTranspose2d as CT
from init_weights import init_weights
import numpy as np



class gen_64(nn.Module):
    
    def __init__(self, 
                 latent_size = 100,
                 complexity = 64,
                 n_col = 3):
        
        super(gen_64, self).__init__()
        self.main = nn.Sequential(
            # BLOCK 0: 1 --> 4
            nn.ConvTranspose2d(latent_size, complexity*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(complexity*8),
            nn.ReLU(True),
            # BLOCK 1: 4 --> 8
            nn.ConvTranspose2d(complexity*8, complexity*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity*4),
            nn.ReLU(True),
            # BLOCK 2: 8 --> 16
            nn.ConvTranspose2d(complexity*4, complexity*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity*2),
            nn.ReLU(True),
            # BLOCK 3: 16 --> 32
            nn.ConvTranspose2d(complexity*2, complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity),
            nn.ReLU(True),
            # BLOCK 4: 32 --> 64
            nn.ConvTranspose2d(complexity, n_col, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.main.apply(init_weights)
        self.main.cuda()

    def forward(self, input):
        output = self.main(input)
        return output

    
    
class Gen(nn.Module):
    
    def __init__(self, 
                 latent_size = 100,
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3):
        
        super(Gen, self).__init__()
        
        self.n_blocks = int(np.log2(64) - 1)
        self.main = nn.Sequential()
        
        # BLOCK 0
        self.main.add_module('b00', CT(latent_size, complexity*8, 4, 1, 0, bias=False))
        self.main.add_module('b01', nn.BatchNorm2d(complexity*8))
        self.main.add_module('b02', nn.ReLU(True))

        # BLOCKS 1 - N-1
        for b in reversed(range(1, self.n_blocks - 1)):
            c_in  = complexity * 2**(b)
            c_out = complexity * 2**(b-1)
            n = 'b' + str(b)
            self.main.add_module(n+'0', CT(c_in, c_out, 4, 2, 1, bias=False))
            self.main.add_module(n+'1', nn.BatchNorm2d(c_out))
            self.main.add_module(n+'2', nn.ReLU(True))
        
        # BLOCK N: 4 --> 1
        n = 'b' + str(self.n_blocks-1)
        self.main.add_module(n+"0", CT(complexity, n_col, 4, 2, 1, bias=False))
        self.main.add_module(n+"1", nn.Tanh())
       
    
    def forward(self, x):
        x = self.main(x)
        return x
    
    