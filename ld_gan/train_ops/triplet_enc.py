
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from ld_gan.utils.model_handler import apply_model, apply_models
from ld_gan.data_proc.transformer import np_to_tensor, tensor_to_np
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import pairwise_distances


    
class TripletEnc:
    
    def __init__(self, 
                 enc, 
                 gen,
                 dis,
                 imgs,
                 lr,
                 n_interpol, 
                 n_pos = 10,
                 n_neg = 10,
                 quantile_pos = 0.05,
                 quantile_neg = 0.3,
                 n_anchors = None,
                 mode = 'min'):
        
        self.criterion = nn.TripletMarginLoss(margin=0.2, p=2)
        self.criterion.cuda()
        
        self.imgs = imgs
        
        self.enc = enc
        self.gen = gen
        self.dis = dis
        self.tnet = Tripletnet(self.enc)
        
        self.n_interpol = n_interpol
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_anchors = n_anchors
        self.quantile_pos = quantile_pos
        self.quantile_neg = quantile_neg
        self.mode = mode
        
        self.opt_enc = optim.Adam(self.tnet.parameters(), lr=lr)
        
    
    def train(self, X, Y, Z, Z_bar):

        # prepare data
        if self.n_anchors is None:
            anchors = tensor_to_np(X)
        else:
            anchors = tensor_to_np(X)[:self.n_anchors]
        
        anchors, pos, neg = generate_triplets(self.enc,
                                              self.gen, 
                                              self.dis, 
                                              anchors, 
                                              self.imgs, 
                                              self.n_interpol, 
                                              n_pos = self.n_pos,
                                              n_neg = self.n_neg,
                                              quantile_pos = self.quantile_pos, 
                                              quantile_neg = self.quantile_pos,
                                              mode = self.mode)

        anchors, pos, neg = np_to_tensor(anchors, pos, neg)
        
        dista, distb, embedded_x, embedded_y, embedded_z = self.tnet(anchors, 
                                                                     pos, 
                                                                     neg)
        
        target = torch.FloatTensor(dista.size()).fill_(1)
        target = target.cuda()
        target = Variable(target)
        
        loss_triplet = self.criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # train
        self.enc.zero_grad()
        loss.backward()
        self.opt_enc.step()
        
        return loss.cpu().data.numpy()[0]
    

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        
        embedded_x = torch.squeeze(embedded_x)
        embedded_y = torch.squeeze(embedded_x)
        embedded_z = torch.squeeze(embedded_x)
        
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
    
            
from time import time
def generate_triplets(enc,
                      gen, 
                      dis, 
                      anchors_x, 
                      imgs,
                      n_interpol, 
                      mode = 'min',
                      n_pos = 10, 
                      n_neg = 10, 
                      quantile_pos=0.05, 
                      quantile_neg=0.3):
    
    #######################################
    # 1. prepare z vectors
    #######################################
    
    # map images to latent space
    x_all = imgs.copy()
    z_all = apply_models(x_all, None, enc)
    anchors = apply_models(anchors_x, None, enc)
    
    n_ancs     = anchors.shape[0]
    n_features = anchors.shape[1]
    n_candidates = n_pos + n_neg
    zs_shape   = (n_interpol, n_ancs, n_candidates, n_features)
    ds_shape   = (n_interpol, n_ancs, n_candidates)
    
    n_can_total = n_ancs * n_candidates
    x_all, z_all = shuffle(x_all, z_all)
    
    # select candidates according to enc-space suggestion
    idxs_pos, idxs_neg = get_enc_space_suggestion(anchors, 
                                                  z_all,
                                                  n_pos=n_pos, 
                                                  n_neg=n_neg, 
                                                  quantile_pos=quantile_pos, 
                                                  quantile_neg=quantile_neg)
    idxs = np.concatenate([idxs_pos, idxs_neg], 1)
    candidates = z_all[idxs]
    x_candidates = x_all[idxs]
        
    zs = np.zeros(zs_shape)
    for i1 in range(n_interpol):
        for i2 in range(n_ancs):
            z_anc = anchors[i2]
            z_can = candidates[i2]
            z_anc = np.tile(z_anc, (n_candidates, 1))
            f1 = i1/float(n_interpol-1)
            f2 = 1 - i1/float(n_interpol-1)
            zs[i1, i2] = f1*z_anc + f2*z_can
            
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
    
    #zs_pos = np.array([c[i] for c, i in zip(candidates, idxs_pos)])
    #zs_neg = np.array([c[i] for c, i in zip(candidates, idxs_neg)])
       
    xs_pos = np.array([c[i] for c, i in zip(x_candidates, idxs_pos)])
    xs_neg = np.array([c[i] for c, i in zip(x_candidates, idxs_neg)])
        
    return anchors_x, xs_pos, xs_neg




def get_enc_space_suggestion(z_anc, 
                             z_all, 
                             n_pos = 10,
                             n_neg = 10,
                             quantile_pos = 0.05, 
                             quantile_neg = 0.3):
    
    n_all = len(z_all)
    n_anc = len(z_anc)
    n_pre_pos = int(quantile_pos * n_all)
    n_pre_neg = int((1.-quantile_neg) * n_all)
        
    # get distances in encoded space
    dists = pairwise_distances(z_anc, z_all)
    idxs_sorted = np.array([d.argsort() for d in dists])
    
    # get indexes fullfilling quantile constraints
    idxs_pos = idxs_sorted[:, 1:n_pre_pos]
    idxs_neg = idxs_sorted[:, -n_pre_neg:]
    
    # random selection for suggestion
    idxs_pos = [i[np.random.randint(0, n_pre_pos-1, n_pos)] for i in idxs_pos]
    idxs_neg = [i[np.random.randint(0, n_pre_neg-1, n_neg)] for i in idxs_neg]
    idxs_pos = np.array(idxs_pos)
    idxs_neg = np.array(idxs_neg)
    
    return idxs_pos, idxs_neg



def get_dis_score_from_z(z_anc, z_can, gen, dis, n_interpol=7, mode='min'):
    
    n_features = z_anc.shape[1]
    n_candidates = len(z_can)
    zs_shape   = (n_interpol, n_candidates, n_features)
    ds_shape   = (n_interpol, n_candidates)
    
    z_anc = np.tile(z_anc, (n_candidates, 1))
    zs = np.zeros(zs_shape)
    for i in range(n_interpol):
        f1 = i/float(n_interpol-1)
        f2 = 1 - i/float(n_interpol-1)
        zs[i] = f1*z_anc + f2*z_can
    
    zs = zs.reshape(-1, n_features)
    
    ds = apply_models(zs, 500, gen, dis)
    ds = ds.reshape(ds_shape)
    
    #return ds
    
    if mode == 'min':
        ds = np.min(ds, axis=1)
    elif mode == 'mean':
        ds = np.mean(ds, axis=1)
    
    return ds

    
def get_dis_score_from_x(x_anc, x_candidates, enc, gen, dis):
    pass

