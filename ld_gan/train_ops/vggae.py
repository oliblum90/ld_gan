import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import __main__ as main
import os
    
class VGGAutoEnc:
    
    def __init__(self, vgg, enc, gen, lr, write_log=True, freq=1):
        
        self.freq = freq
        
        self.criterion = nn.L1Loss()
        # self.criterion.size_average = False
        self.criterion.cuda()
        
        self.vgg = vgg
        self.enc = enc
        self.gen = gen
        
        self.opt_enc = optim.Adam(self.enc.parameters(), lr=lr)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr)
        
        self.write_log = write_log
        self.log_fname = os.path.join("projects", 
                                      main.__file__, 
                                      "log", 
                                      "vgg_logs.txt")
        
        
    def _init_log(self):
        header = "e0 e1 e2 e3 e4 e5"
        with open(self.log_fname, 'w') as f:
            f.write(header)
            
    
    def train(self, X, Y, Z, Z_bar):

        bs = X.size(0)

        self.enc.train()
        self.gen.train()
        
        self.gen.zero_grad()
        self.enc.zero_grad()
        C1, C2, C3, C4, C5 = self.vgg(X)
        z = self.enc(X)
        x = self.gen(z)
        c1, c2, c3, c4, c5 = self.vgg(x.detach())
        e0 = self.criterion(x, X)
        e1 = self.criterion(c1, C1.detach())
        e2 = self.criterion(c2, C2.detach())
        e3 = self.criterion(c3, C3.detach())
        e4 = self.criterion(c4, C4.detach())
        e5 = self.criterion(c5, C5.detach())
        err = e0 + e1 + e2 + e3 + e4 + e5
        err.backward()
        mean_x = x.data.mean()
        self.opt_enc.step()
        self.opt_gen.step()
        
        self.enc.eval()
        self.gen.eval()

        # write log
        if self.write_log:
            
            if os.path.isfile(self.log_fname) == False:
                self._init_log()
                
            losses = [e0.cpu().data.numpy()[0],
                      e1.cpu().data.numpy()[0],
                      e2.cpu().data.numpy()[0],
                      e3.cpu().data.numpy()[0],
                      e4.cpu().data.numpy()[0],
                      e5.cpu().data.numpy()[0]]
            line = " ".join([str(l) for l in losses])
            with open(self.log_fname, 'a') as f:
                f.write("\n" + line)
        
        return err.cpu().data.numpy()[0]
    
    
    
    
    