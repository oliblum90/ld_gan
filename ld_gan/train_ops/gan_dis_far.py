import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from ld_gan.utils.model_handler import apply_models
from ld_gan.data_proc.transformer import np_to_tensor, tensor_to_np


def ones(bs):
    return Variable(torch.from_numpy(np.ones(bs))).cuda().float()
def zeros(bs):
    return Variable(torch.from_numpy(np.zeros(bs))).cuda().float()

    
class GanDisFar:
    
    def __init__(self, enc, gen, dis, lr, imgs, batch_size = None, freq=1):
        
        self.freq = freq
        
        self.criterion = nn.BCELoss()
        self.criterion.cuda()
        
        self.enc = enc
        self.gen = gen
        self.dis = dis
        
        self.imgs = imgs
        self.batch_size = batch_size
        
        self.opt_dis = optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999))
        
    
    def train(self, X, Y, Z, Z_bar):

        if self.batch_size is None:
            self.batch_size = X.size(0)

        # generate random samples
        idxs_1 = np.random.randint(0, len(self.imgs), size=self.batch_size)
        idxs_2 = np.random.randint(0, len(self.imgs), size=self.batch_size)
        imgs_1 = self.imgs[idxs_1]
        imgs_2 = self.imgs[idxs_2]
        z1 = apply_models(imgs_1, None, self.enc)
        z2 = apply_models(imgs_2, None, self.enc)
        z = (z1 + z2) / 0.5
        z = np_to_tensor(z)
        x = np_to_tensor(imgs_1)
        
        self.dis.train()
        
        self.dis.zero_grad()
        d = self.dis(x)
        errD_real = self.criterion(d, ones(self.batch_size))
        errD_real.backward()
        D_x = d.data.mean()

        # train with fake
        x = self.gen(z)
        d = self.dis(x.detach())
        errD_fake = self.criterion(d, zeros(self.batch_size))
        errD_fake.backward()
        D_G_z1 = d.data.mean()
        errD = errD_real + errD_fake
        self.opt_dis.step()
        
        self.dis.eval()
        
        return errD.cpu().data.numpy()[0]
    