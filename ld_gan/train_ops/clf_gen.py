import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import __main__ as main
import ld_gan
from scipy import stats


def ones(bs):
    return Variable(torch.from_numpy(np.ones(bs))).cuda().float()
def zeros(bs):
    return Variable(torch.from_numpy(np.zeros(bs))).cuda().float()

    
class GenCLF:
    
    def __init__(self, gen, enc, clf_layer, Y_all, lr, 
                 freq=1, write_log=True, mode="orig_y"):
        
        self.freq = freq
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        
        self.gen = gen
        self.enc = enc
        self.clf_layer = clf_layer
        
        self.Y_all = Y_all
        self.mode = mode
        
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr)
        
        self.write_log = write_log
        self.log_fname = os.path.join("projects", 
                                      main.__file__, 
                                      "log", 
                                      "gclf_acc.txt")
        
    def _init_log(self):
        header = "gclf_acc"
        with open(self.log_fname, 'w') as f:
            f.write(header)
    
    
    def train(self, X, Y, Z, batch_idxs, nn_idxs, sr_idxs, z_all):

        self.gen.zero_grad()
        
        x = self.gen(Z)
        z = self.enc(x)
        y = self.clf_layer(z)
        
        
        if self.mode == "mode":
            for i in range(len(Y)):
                ys = self.Y_all[nn_idxs[i]]
                Y[i] = stats.mode(ys)[0][0]
        
        err = self.criterion(y, Y)
        
        err.backward()
        
        self.opt_gen.step()
        
        
        # write log
        if self.write_log:
            
            if os.path.isfile(self.log_fname) == False:
                self._init_log()
                
            y = ld_gan.data_proc.transformer.tensor_to_np(y)
            Y = ld_gan.data_proc.transformer.tensor_to_np(Y)
            y = np.argmax(y, axis = 1)
            acc = float((Y == y).sum()) / len(Y)
                
            line = str(acc)
            with open(self.log_fname, 'a') as f:
                f.write("\n" + line)
        
        
        return err.cpu().data.numpy()[0]
    
    