import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from ld_gan.models import init_weights


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
    
    def __init__(self, enc, lr, n_features, n_classes):
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        
        self.enc = enc
        self.clf_layer = _ClfLayer(n_features, n_classes)
        
        self.opt_enc = optim.Adam(self.enc.parameters(), lr=lr)
        self.opt_clf_layer = optim.Adam(self.clf_layer.parameters(), lr=lr)

    
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
        
        return err.cpu().data.numpy()[0]
    
    
    
    
    