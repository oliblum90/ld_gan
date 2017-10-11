import os
import ld_gan
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import scipy.misc


def nn(z_batch, z_all, n_neighbors=1):
    dists = pairwise_distances(z_batch, z_all)
    idxs = np.argsort(dists, axis=1)[:, :n_neighbors]
    return idxs


def create_nn_imgs(X, Xt, x, path):
    
    X_flat = X.reshape((X.shape[0], -1))
    Xt_flat = Xt.reshape((Xt.shape[0], -1))
    x_flat = x.reshape((x.shape[0], -1))
    
    print "compute nearest neighbors..."
    idxs_real = nn(Xt_flat, X_flat)[:,0]
    idxs_fake = nn(Xt_flat, x_flat)[:,0]
    
    print "save imgs..."
    for idx in tqdm(range(len(Xt))):
        i1, i2, i3 = Xt[idx], x[idxs_fake[idx]], X[idxs_real[idx]]
        img = np.concatenate([i1, i2, i3], axis=1)
        fname = os.path.join(path, str(idx).zfill(3)+".png")
        scipy.misc.imsave(fname, img)