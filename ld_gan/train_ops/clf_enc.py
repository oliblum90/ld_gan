import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from ld_gan.models import init_weights
import ld_gan
import __main__ as main
import os


class _ClfLayer(nn.Module):
    
    def __init__(self, n_features, n_classes):
        
        super(_ClfLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_features, n_classes),
        )
        self.main.apply(init_weights)
        self.main.cuda()
        
        self.n_features = n_features
        self.n_classes = n_classes

    def forward(self, input):
        input = input.view(-1, self.n_features)
        output = self.main(input)
        return output
    
    
class Clf:
    
    def __init__(self, enc, lr, n_features, n_classes, write_log=True):
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        
        self.enc = enc
        self.clf_layer = _ClfLayer(n_features, n_classes)
        
        self.opt_enc = optim.Adam(self.enc.parameters(), lr=lr)
        self.opt_clf_layer = optim.Adam(self.clf_layer.parameters(), lr=lr)
        #self.opt_enc = optim.SGD(self.enc.parameters(), lr=lr, momentum=0.9)
        #self.opt_clf_layer = optim.SGD(self.clf_layer.parameters(), lr=lr, momentum=0.9)
        
        self.write_log = write_log
        self.log_fname = os.path.join("projects", 
                                      main.__file__, 
                                      "log", 
                                      "clf_acc.txt")

        
    def _init_log(self):
        header = "clf_acc"
        with open(self.log_fname, 'w') as f:
            f.write(header)
        
    
    def train(self, X, Y, Z, Z_bar):
        
        self.enc.zero_grad()
        self.clf_layer.zero_grad()
        
        z = self.enc(X)
        y = self.clf_layer(z)
        
        err = self.criterion(y, Y)
        err.backward()
        mean = y.data.mean()
        
        self.opt_enc.step()
        self.opt_clf_layer.step()
        
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
    
    
    
    
    