
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from ld_gan.utils.model_handler import apply_model, apply_models
from ld_gan.data_proc.transformer import np_to_tensor, tensor_to_np
import torch.nn.functional as F
from sklearn.utils import shuffle


    
class TripletEnc:
    
    def __init__(self, 
                 enc, 
                 gen,
                 dis,
                 imgs,
                 lr,
                 margin,
                 p,
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
        
    
    def train(self, X, Y, Z, Z_bar):

        # prepare data
        anchors = tensor_to_np(X)
        
        anchors, pos, neg = generate_triplets(self.enc,
                                              self.gen, 
                                              self.dis, 
                                              anchors, 
                                              self.imgs, 
                                              self.n_interpol, 
                                              self.n_candidates,
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
    
    
def generate_triplets(enc,
                      gen, 
                      dis, 
                      anchors_x, 
                      x_all,
                      n_interpol, 
                      n_candidates,
                      mode = 'min'):
    
    #######################################
    # 1. prepare z vectors
    #######################################
    
    # map images to latent space
    z_all = apply_models(x_all, None, enc)
    anchors = apply_models(anchors_x, None, enc)
    
    n_ancs     = anchors.shape[0]
    n_features = anchors.shape[1]
    zs_shape   = (n_interpol, n_ancs, n_candidates, n_features)
    ds_shape   = (n_interpol, n_ancs, n_candidates)
    
    n_can_total = n_ancs * n_candidates
    x_all, z_all = shuffle(x_all, z_all)
    z_all = np.tile(z_all, (n_can_total / len(z_all) + 1, 1))[:n_can_total]
    x_all = np.tile(x_all, (n_can_total / len(x_all) + 1, 1, 1, 1))[:n_can_total]
    
    candidates = np.split(z_all, n_ancs)
    candidates_x = np.split(x_all, n_ancs)
    
    zs = np.zeros(zs_shape)
    for i1 in range(n_interpol):
        for i2 in range(n_ancs):
            z_anc = anchors[i2]
            z_can = candidates[i2]
            z_anc = np.tile(z_anc, (n_candidates, 1))
            f1 = i1/float(n_interpol-1)
            f2 = 1 - i1/float(n_interpol-1)
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
    
    #zs_pos = np.array([c[i] for c, i in zip(candidates, idxs_pos)])
    #zs_neg = np.array([c[i] for c, i in zip(candidates, idxs_neg)])
       
    xs_pos = np.array([c[i] for c, i in zip(candidates_x, idxs_pos)])
    xs_neg = np.array([c[i] for c, i in zip(candidates_x, idxs_neg)])
        
    return anchors_x, xs_pos, xs_neg
            
            