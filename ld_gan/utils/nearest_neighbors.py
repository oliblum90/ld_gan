import numpy as np
import torch 
from torch.autograd import Variable
import torch.nn.functional as F

def nn_gpu(z_all, z_batch, n_neighbors = 5):
    
    if type(z_all) is np.ndarray:
        type_np = True
        z_all = Variable(torch.from_numpy(z_all).cuda())
        z_batch = Variable(torch.from_numpy(z_batch).cuda())
    else:
        type_np = False
        
    z_all = F.normalize(z_all, dim=1)
    z_batch = F.normalize(z_batch, dim=1)
    dists = torch.mm(z_batch, z_all.permute(1, 0))
    _, idxs = torch.sort(dists, 1)
    
    if n_neighbors is not None:
        idxs = idxs[:, -n_neighbors:]

    # reverse idxs
    inv_idx = torch.arange(idxs.size(0)-1, -1, -1).long().cuda()
    idxs = idxs[inv_idx]
    
    if type_np:
        idxs = (idxs.data).cpu().numpy()
    
    return idxs