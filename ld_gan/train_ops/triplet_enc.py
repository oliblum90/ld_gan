
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from ld_gan.utils.model_handler import apply_model, apply_models
from tripletnet import Tripletnet

    
class TripletEnc:
    
    def __init__(self, 
                 enc, 
                 gen,
                 dis,
                 imgs,
                 lr,
                 margin,
                 p,
                 z_all, 
                 n_interpol, 
                 n_candidates,
                 mode = 'min'):
        
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)
        self.criterion.cuda()
        
        self.imgs = imgs
        
        self.enc = enc
        self.gen = gen
        self.dis = dis
        self.tnet = Tripletnet(self.enc)
        
        self.n_interpol = n_interpol
        self.n_candidates = n_candidates
        self.mode = mode
        
        self.opt_enc = optim.Adam(self.tnet.parameters(), lr=lr)
        
    
    # Z are the chosen anchors
    def train(self, X, Y, Z, Z_bar):

        # prepare data
        z_all = z_enc = apply_model(self.enc, self.imgs)
        anchors = tensor_to_np(Z)
        anchors, zs_pos, zs_neg = generate_triplets(self.gen, 
                                                    self.dis, 
                                                    anchors, 
                                                    z_all, 
                                                    self.n_interpol, 
                                                    self.n_candidates,
                                                    mode = self.mode)

        anchors, zs_pos, zs_neg = np_to_tensor(anchors, zs_pos, zs_neg)
        
        # train
        err = self.criterion(anchors, zs_pos, zs_neg)
        err.backward()
        mean_x = x.data.mean()
        self.opt_enc.step()
        
        return err.cpu().data.numpy()[0]
    
    
    
def generate_triplets(gen, 
                      dis, 
                      anchors, 
                      z_all, 
                      n_interpol, 
                      n_candidates,
                      mode = 'min'):
    
    #######################################
    # 1. prepare z vectors
    #######################################
    
    n_ancs     = anchors.shape[0]
    n_features = anchors.shape[1]
    zs_shape   = (n_interpol, n_ancs, n_candidates, n_features)
    ds_shape   = (n_interpol, n_ancs, n_candidates)
    
    n_can_total = n_ancs * n_candidates
    z_all = np.tile(z_all, (n_can_total / len(z_all) + 1, 1))[:n_can_total]
    candidates = np.split(z_all, n_ancs)
    
    zs = np.zeros(zs_shape)
    for i1 in range(n_interpol):
        for i2 in range(n_ancs):
            z_anc = anchors[i2]
            z_can = candidates[i2]
            z_anc = np.tile(z_anc, (n_candidates, 1))
            f1 = i1/float(N_INTERPOL-1)
            f2 = 1 - i1/float(N_INTERPOL-1)
            zs[i1, i2] = f1*z_anc + f2*z_anc
            
    zs = zs.reshape(-1, n_features)
            
        
    #######################################
    # 2. get d-score for z vectors
    #######################################
    
    ds = apply_models(zs, 3000, gen, dis)
    ds = ds.reshape(ds_shape)

    
    #######################################
    # 3. pos / neg sample for each anchor
    #######################################
    
    if mode == 'min':
        ds = np.min(ds, axis=0)
    elif mode == 'mean':
        ds = np.mean(ds, axis=0)
    
    idxs_pos = np.argmax(ds, axis=1)
    idxs_neg = np.argmin(ds, axis=1)
    
    zs_pos = np.array([c[i] for c, i in zip(candidates, idxs_pos)])
    zs_neg = np.array([c[i] for c, i in zip(candidates, idxs_neg)])
    
        
    return anchors, zs_pos, zs_neg
            
            