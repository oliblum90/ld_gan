import ld_gan
import torch.optim as optim
import torch

def entropy_loss(z_all, z_batch, bw=4.5):
    
    z_all = torch.squeeze(z_all)
    z_batch = torch.squeeze(z_batch)
    
    # get cosine distances
    dist_mat = torch.mm(z_all, z_batch.t())
    dist_norm = (z_batch * z_batch).sum(1)
    dist_mat = dist_norm - dist_mat

    # get gaussian transform of distances
    gaus_mat = torch.exp(- dist_mat / bw)

    # get error
    err = torch.mean(gaus_mat)
    
    return err


class EntropyEnc:
    
    def __init__(self, enc, lr, imgs, freq=1, bw = 4.5):
        
        self.freq = freq
        self.z_enc = ld_gan.utils.model_handler.apply_model(enc, imgs, batch_size=200)
        self.z_enc = ld_gan.data_proc.transform(self.z_enc)
        self.z_enc = torch.squeeze(self.z_enc)
        self.bw = bw
        self.enc = enc
        self.opt_enc = optim.Adam(self.enc.parameters(), lr=lr, betas=(0.5, 0.999))
        
    
    def train(self, X, Y, Z, Z_bar):
        
        self.enc.zero_grad()
        z = self.enc(X)
        err = entropy_loss(self.z_enc.detach(), z)
        err.backward()
        self.opt_enc.step()
        
        return err.cpu().data.numpy()