import os
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
import __main__ as main
import ld_gan


    
class TripletEnc:
    
    def __init__(self, 
                 enc, gen, dis,
                 imgs,
                 lr,
                 n_interpol = 10, 
                 quantile_pos = 0.1,
                 quantile_neg = 0.1,
                 mode = 'min', 
                 freq=1,
                 logging = True,
                 visualize = False):
        
        self.freq = freq
        
        self.criterion = nn.TripletMarginLoss(margin=0.2, p=2)
        self.criterion.cuda()
        
        self.imgs = imgs
        
        self.enc = enc
        self.gen = gen
        self.dis = dis
        self.tnet = Tripletnet(self.enc)
        
        self.n_interpol = n_interpol
        self.quantile_pos = quantile_pos
        self.quantile_neg = quantile_neg
        self.mode = mode
        self.logging = logging
        self.visualize = visualize
        
        self.opt_enc = optim.Adam(self.tnet.parameters(), lr=lr)
        
        if logging:
            self.log_fname = os.path.join("projects", 
                                          main.__file__, 
                                          "log", 
                                          "dis_score.txt")
        
        
    def _init_log(self):
        header = "mean min max"
        with open(self.log_fname, 'w') as f:
            f.write(header)
            
            
    def train(self, X, Y, Z, batch_idxs, nn_idxs, sr_idxs, z_all):
        
        # del own idx
        sr_idxs = np.array([si[si!=bi] for bi, si in zip(batch_idxs, sr_idxs)])

        #######################################
        # 1. prepare z vectors
        #######################################
        
        zs_anc = z_all[batch_idxs]
        zs_potential = z_all[sr_idxs]

        n_interpol = self.n_interpol
        batch_size = zs_potential.shape[0]
        n_sr_zs    = zs_potential.shape[1]
        n_features = zs_potential.shape[2]

        zs_paths = np.zeros((n_interpol, batch_size, n_sr_zs, n_features))
        for i1 in range(n_interpol):
            for i2 in range(batch_size):
                for i3 in range(n_sr_zs):
                    f1 = (i1+1)/float(n_interpol+1)
                    f2 = 1 - f1
                    zs_paths[i1, i2, i3, :] = f1*zs_anc[i2] + f2*zs_potential[i2, i3]
                    
        zs = zs_paths.reshape(-1, n_features)
        
        
        #######################################
        # 2. get d-score for z vectors
        #######################################
    
        ds = apply_models(zs, 1000, self.gen, self.dis)
        ds = ds.reshape((n_interpol, batch_size, n_sr_zs))
        
        
        #######################################
        # 3. pos / neg sample for each anchor
        #######################################

        if self.mode == 'min':
            ds = np.min(ds, axis=0)
        elif self.mode == 'mean':
            ds = np.mean(ds, axis=0)
        
        sr_idxs_sorted = np.zeros_like(sr_idxs)
        for i1 in range(batch_size):
            sorted_args = np.argsort(-ds[i1])
            sr_idxs_sorted[i1] = (sr_idxs[i1])[sorted_args]
                
        n_pos = int(self.quantile_pos * n_sr_zs)
        n_neg = int(self.quantile_neg * n_sr_zs)
        idxs_pos = sr_idxs_sorted[:, :n_pos]
        idxs_neg = sr_idxs_sorted[:, -n_neg:]
        idx_pos = np.array([i[np.random.randint(0, n_pos)] for i in idxs_pos])
        idx_neg = np.array([i[np.random.randint(0, n_neg)] for i in idxs_neg])
        
        x_pos = self.imgs[idx_pos]
        x_neg = self.imgs[idx_neg]
        x_anc = self.imgs[batch_idxs]
        
        
        #######################################
        # 4. Visualize first score distribution
        #######################################
        
        if self.visualize:
            
            print "anc / pos / neg"
            ld_gan.visualize.disp([x_anc[0], x_pos[0], x_neg[0]])
            
            print "all sorted"
            sorted_args = np.argsort(-ds[0])
            abs_args = (sr_idxs[0])[sorted_args]
            sorted_imgs = self.imgs[abs_args]
            ld_gan.visualize.disp_array(sorted_imgs, (5, 10))
                        
            print "compare"
            zs_anc = z_all[batch_idxs][0] # (64, 64, 3)
            zs_potential = z_all[sr_idxs][0] # (50, 64, 64, 3)
            xs_potential = self.imgs[sr_idxs][0] # (50, 64, 64, 3)
            
            # compute d of potentials reagrding ancer
            ds_ipol_list = []
            for i1 in range(n_sr_zs):
                zs_ipol = []
                for i2 in range(n_interpol):
                    f1 = (i2+1)/float(n_interpol+1)
                    f2 = 1 - f1
                    zs_ipol.append(f1*zs_anc + f2*zs_potential[i1])
                zs_ipol = np.array(zs_ipol)
                xs_ipol = apply_models(zs_ipol, None, self.gen)
                ds_ipol = apply_models(xs_ipol, None, self.dis)
                ds_ipol_list.append(np.min(ds_ipol))
                #print np.min(ds_ipol)
                #ld_gan.visualize.disp(xs_ipol)
                
            sorted_args = np.argsort(-np.array(ds_ipol_list))
            sorted_imgs = xs_potential[sorted_args]
            ld_gan.visualize.disp_array(sorted_imgs, (5, 10))
                
            
        #######################################
        # 5. write log
        #######################################
        
        if self.logging:            
            if os.path.isfile(self.log_fname) == False:
                self._init_log()
            ds_mean = ds.mean()
            ds_min = np.min(ds, axis=1).mean()
            ds_max = np.max(ds, axis=1).mean()
            line = str(ds_mean) + ' ' + \
                   str(ds_min) + ' ' + \
                   str(ds_max)
            with open(self.log_fname, 'a') as f:
                f.write("\n" + line)
        
        
        #######################################
        # 6. Train
        #######################################
        
        x_anc, x_pos, x_neg = np_to_tensor(x_anc, x_pos, x_neg)
        
        dista, distb, embedded_x, embedded_y, embedded_z = self.tnet(x_anc, x_pos, x_neg)
                
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