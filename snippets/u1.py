import os
os.chdir("../")

import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan

RAND_SEED = 42
cudnn.benchmark = True
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed_all(RAND_SEED)




import os
os.chdir("../")

import ld_gan
import torch.nn as nn
import numpy as np


X, Y, Xt, Yt = ld_gan.data_proc.data_loader.load_data(1, 
                                                      verbose=1, 
                                                      resize = 64,
                                                      split_test_train_ratio = 0.2)
n_classes = Y.shape[1]
Y = np.argmax(Y, axis = 1)
Yt = np.argmax(Yt, axis = 1)



from ld_gan.models import init_weights

class Enc(nn.Module):
    
    def __init__(self, 
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3,
                 n_features = 256,
                 activation = "tanh", 
                 add_clf_layer = True, 
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
            # x = F.softmax(x)
        
        return x
        
        
        
enc = Enc()

enc.cuda()
enc.apply(ld_gan.models.init_weights)


sampler = ld_gan.sample.generate_rand_noise(X, Y, 256, 256)


criterion = nn.CrossEntropyLoss()


from ld_gan.data_proc.transformer import np_to_tensor, tensor_to_np
import torch.optim as optim
import torch.nn.functional as F
import torch


opt = optim.Adam(enc.parameters(), lr=0.001)


errs = []

for i in range(10000):
    
    X, Y, Z, Z_bar = sampler.next()
    X, Y, Z, Z_bar = np_to_tensor(X, Y, Z, Z_bar)
    
    enc.zero_grad()
    y = enc(X)
    err = criterion(y, Y)
    err.backward()
    mean = y.data.mean()

    errs.append(err.cpu().data.numpy()[0])
    
    opt.step()
    
    if i % 100 == 0:
        yt = Yt[:256]
        y = enc(ld_gan.data_proc.transformer.np_to_tensor(Xt[:256]))
        y = ld_gan.data_proc.transformer.tensor_to_np(y)
        y = np.argmax(y, axis = 1)
        acc = float((yt == y).sum()) / len(yt)
        print "\n"
        print "accuracy: ", acc
        print "loss:", np.array(errs).mean()
        errs = []