import torch.nn as nn
from init_weights import init_weights
import torch.nn.functional as F
import numpy as np


class enc_64(nn.Module):
    
    def __init__(self, 
                 complexity = 64,
                 n_col = 3,
                 n_features = 256,
                 activation = "tanh"):
        
        super(enc_64, self).__init__()
        self.main = nn.Sequential(
            # BLOCK 0: 64 --> 32
            nn.Conv2d(n_col, complexity, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 1: 32 --> 16
            nn.Conv2d(complexity, complexity * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 2: 16 --> 8
            nn.Conv2d(complexity * 2, complexity * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 3: 8 --> 4
            nn.Conv2d(complexity * 4, complexity * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 4: 4 --> 1
            nn.Conv2d(complexity * 8, n_features, 4, 1, 0, bias=False),
        )
        self.main.apply(init_weights)
        self.main.cuda()
        
        self.activation = activation

    def forward(self, input):
        output = self.main(input)
        
        if self.activation == "tanh":
            output = F.tanh(output)
            
        return output
    

    
    
class Enc(nn.Module):
    
    def __init__(self, 
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3,
                 n_features = 256,
                 activation = "tanh", 
                 add_clf_layer = False, 
                 n_classes = 102):
        
        super(Enc, self).__init__()
        
        self.activation = activation
        self.add_clf_layer = add_clf_layer
        self.n_blocks = int(np.log2(64) - 1)
        
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
        n = 'b' + str(self.n_blocks-1) + "0"
        self.main.add_module(n, nn.Conv2d(c_out, n_features, 4, 1, 0, bias=False))
        
        # CLF-LAYER
        if self.add_clf_layer:
            self.clf = nn.Linear(n_features, n_classes)

    def forward(self, x):
        
        x = self.main(x)
            
        if self.activation == "tanh":
            x = F.tanh(x)
            
        if self.add_clf_layer:
            x = x.view(x.size(0), x.size(1))
            x = self.clf(x)
            #x = F.softmax(x)
        
        return x
    
    
    
    
    
    
    
    