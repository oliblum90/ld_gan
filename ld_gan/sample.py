"""
every sampler is a iterator which yields 3 outputs:

 - z_in:
   z to be put into generator
   
 - z_out:
   optimal z to be obtained after encoding generated img given z_in
   
 - img
   optimal image to be generated given z_in
   
outputs that can not be used with the sampling method are set to None
"""


import numpy as np
import sklearn.preprocessing
import scipy
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Process, Queue
import ld_gan
from ld_gan.utils.log_time import log_time
from tqdm import tqdm
from ld_gan.utils.nearest_neighbors import nn_gpu



        
def kde_sampler_life(enc, 
                     X, 
                     y, 
                     batch_size, 
                     bandwidth = 0.5, 
                     nn_subset_size = None):
    
    while True:
        
        log_time("get_z_enc")
        if nn_subset_size is None:
            imgs = X
        else:
            rand_idxs = np.random.randint(0, len(X), nn_subset_size)
            imgs = X[rand_idxs]
        z_enc = ld_gan.utils.model_handler.apply_model(enc, imgs, batch_size = 500)
        log_time("get_z_enc")
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        img_batch = imgs[batch_idxs]
        y_batch = y[batch_idxs]
        kde = KernelDensity(bandwidth = bandwidth).fit(z_enc)
        z_batch = kde.sample(batch_size)
        
        yield img_batch, y_batch, z_batch, z_batch
        

def gmm_sampler_life(enc, 
                     X, 
                     y, 
                     batch_size, 
                     n_classes):
    
    while True:
        
        log_time("get_z_enc")
        z_enc = ld_gan.utils.model_handler.apply_model(enc, X, batch_size = 500)
        log_time("get_z_enc")
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        x_batch = X[batch_idxs]
        y_batch = y[batch_idxs]
        gmm = GaussianMixture(n_components = 102, covariance_type='diag').fit(z_enc)
        z_batch = gmm.sample(batch_size)[0]
        
        yield x_batch, y_batch, z_batch, None, None, None
        
        
def nn_sampler_life(enc, 
                    X, 
                    y, 
                    batch_size, 
                    nn_search_radius = 50, 
                    n_neighbors = 5,
                    sub_set_size = None):
    
    while True:
        
        if sub_set_size is not None:
            x = X[np.random.randint(0, len(X), sub_set_size)]
        else:
            x = X
        
        log_time("get_z_enc")
        #z_enc = ld_gan.utils.model_handler.apply_model(enc, x, batch_size=batch_size)
        z_enc = ld_gan.utils.model_handler.apply_models(x, batch_size, enc)
        log_time("get_z_enc")
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        
        log_time("find_nn")
        sr_idxs = nn_gpu(z_enc, z_enc[batch_idxs], n_neighbors=nn_search_radius)
        nn_idxs = [i[np.random.randint(0, nn_search_radius, n_neighbors)] for i in sr_idxs]
        log_time("find_nn")

        # get z_batch
        batch_z_all = z_enc[nn_idxs]
        rand_weights = np.random.rand(n_neighbors, batch_size)
        rand_weights = rand_weights / np.sum(rand_weights, axis=0)
        rand_weights = rand_weights.transpose()
        z_batch = [np.average(za, 0, w) for w, za in zip(rand_weights, batch_z_all)]
        z_batch = np.array(z_batch)
        
        x_batch = x[batch_idxs]
        y_batch = y[batch_idxs]
        z_batch_orig = z_enc[batch_idxs]
        
        yield x_batch, y_batch, z_batch, batch_idxs, nn_idxs, sr_idxs, z_enc

