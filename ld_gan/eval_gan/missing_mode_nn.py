import os
import ld_gan
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import scipy.misc


def nn(z_all, z_batch, batch_size = None):
    if batch_size is None:
        dists = pairwise_distances(z_all, z_batch)
        idxs = np.argsort(dists, axis=1)[:, 0]
    else:
        n_batches = int(len(z_batch) / batch_size)
        batches = np.array_split(z_batch, n_batches)
        dist_list = []
        idxs_list = []
        idx_offset = 0
        dists_list = []
        idxs_list = []
        
        for b in tqdm(batches):
            dists = pairwise_distances(z_all, b)
            idxs = np.argsort(dists, axis=1)[:, 0]
            dists = np.min(dists, axis=1)
            idxs += idx_offset
            idx_offset += len(b)
            idxs_list.append(idxs)
            dists_list.append(dists)
        dists_list = np.array(dists_list)
        idxs_list = np.array(idxs_list)
        
        idxs = idxs_list[np.argmin(dists_list, axis=0), range(idxs_list.shape[1])]
    return idxs


def create_nn_imgs(X, Xt, x, path):
    
    X_flat = X.reshape((X.shape[0], -1))
    Xt_flat = Xt.reshape((Xt.shape[0], -1))
    x_flat = x.reshape((x.shape[0], -1))
    
    print "compute nearest neighbors..."
    idxs_real = nn(Xt_flat, X_flat)
    idxs_fake = nn(Xt_flat, x_flat, batch_size=10000)
    
    print "save imgs..."
    for idx in tqdm(range(len(Xt))):
        i1, i2, i3 = Xt[idx], x[idxs_fake[idx]], X[idxs_real[idx]]
        img = np.concatenate([i1, i2, i3], axis=1)
        fname = os.path.join(path, str(idx).zfill(3)+".png")
        scipy.misc.imsave(fname, img)

