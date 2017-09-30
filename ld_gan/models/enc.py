import torch.nn as nn
from init_weights import init_weights
import torch.nn.functional as F
import numpy as np
import torch
import ld_gan
from sklearn.neighbors import NearestNeighbors

    
class Enc(nn.Module):
    
    def __init__(self, 
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3,
                 n_features = 256,
                 activation = "tanh",
                 w_norm     = False,
                 batchnorm  = True):
        
        super(Enc, self).__init__()
        
        if w_norm:
            from WeightNormalizedConv import WeightNormalizedConv2d as Conv2d
        else:
            from torch.nn import Conv2d

        self.activation = activation
        self.n_blocks = int(np.log2(ipt_size) - 1)
        
        self.main = nn.Sequential()
        
        # BLOCK 0
        self.main.add_module('b00', Conv2d(n_col, complexity, 4, 2, 1, bias=False))
        self.main.add_module('b01', nn.LeakyReLU(0.2, inplace=True))

        # BLOCKS 1 - N-1
        for b in range(1, self.n_blocks - 1):
            c_in  = complexity * 2**(b-1)
            c_out = complexity * 2**(b)
            n = 'b' + str(b)
            self.main.add_module(n+'0', Conv2d(c_in, c_out, 4, 2, 1, bias=False))
            if batchnorm:
                self.main.add_module(n+'1', nn.BatchNorm2d(c_out))
            self.main.add_module(n+'2', nn.LeakyReLU(0.2, inplace=True))
        
        # BLOCK N: 4 --> 1
        n = 'b' + str(self.n_blocks-1) + "0"
        self.main.add_module(n, Conv2d(c_out, n_features, 4, 1, 0, bias=False))
        

    def forward(self, x):
        
        x = self.main(x)
            
        if self.activation == "tanh":
            x = F.tanh(x)
        elif self.activation == "auto_unify":
            x = (x - x.mean()) / x.std()
            x = 1. / (torch.exp(-(358. * x)/23. + 111. * torch.atan(37. * x / 294.)) + 1.)
        
        return x
    
    
class VggFeatureEnc(nn.Module):
    
    def __init__(self, 
                 vgg_features,
                 imgs,
                 ipt_size   = 64,
                 complexity = 64,
                 n_col      = 3,
                 n_features = 256,
                 activation = "tanh"):
        
        super(VggFeatureEnc, self).__init__()
        
        self.activation = activation
        
        self.vgg_features = ld_gan.data_proc.transformer.np_to_tensor(vgg_features)
        self.vgg_features = self.vgg_features.squeeze()
        self.vgg_features.cuda()
        self.ref_ids = get_img_id(imgs)
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.ref_ids)
        
        self.main = nn.Sequential()

        self.main.add_module('l1', nn.Linear(4096, 1024))
        self.main.add_module('l2', nn.ReLU())
        self.main.add_module('l3', nn.Linear(1024, 1024))
        self.main.add_module('l4', nn.ReLU())
        self.main.add_module('l5', nn.Linear(1024, n_features))
        

    def forward(self, x):
        
        imgs_batch = ld_gan.data_proc.transformer.tensor_to_np(x)
        _, idxs = self.nn.kneighbors(get_img_id(imgs_batch))
        idxs = torch.cuda.LongTensor(np.squeeze(idxs)).cuda()
        x = self.vgg_features[idxs]
        x = x.cuda()
        x = self.main(x)
            
        if self.activation == "tanh":
            x = F.tanh(x)
        elif self.activation == "auto_unify":
            x = (x - x.mean()) / x.std()
            x = 1. / (torch.exp(-(358. * x)/23. + 111. * torch.atan(37. * x / 294.)) + 1.)
        
        x = x.unsqueeze(2).unsqueeze(2)
        
        return x
    
    
def get_img_id(img, 
               base_pts = [[31, 31], [0, 0], [63, 0], [63, 0], [63, 63]]):
    ids = []
    for pts in base_pts:
        ids.append(img[:, pts[0], pts[1], 0])
        ids.append(img[:, pts[0], pts[1], 1])
        ids.append(img[:, pts[0], pts[1], 2])
    return np.array(ids).transpose()

