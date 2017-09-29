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
        
        
        
def nn_sampler_life(enc, 
                    X, 
                    y, 
                    batch_size, 
                    nn_search_radius = 50, 
                    n_neighbors = 5, 
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
        
        log_time("find_nn")
        idxs = nn_gpu(z_enc, z_enc[batch_idxs], n_neighbors=nn_search_radius)
        idxs = [i[np.random.randint(0, nn_search_radius, n_neighbors)] for i in idxs]
        log_time("find_nn")

        # get z_batch
        batch_z_all = z_enc[idxs]
        rand_weights = np.random.rand(n_neighbors, batch_size)
        rand_weights = rand_weights / np.sum(rand_weights, axis=0)
        rand_weights = rand_weights.transpose()
        z_batch = [np.average(z_all, 0, w) for w, z_all in zip(rand_weights, batch_z_all)]
        z_batch = np.array(z_batch)
        
        img_batch = imgs[batch_idxs]
        
        y_batch = y[batch_idxs]
        
        yield img_batch, y_batch, z_batch, batch_idxs, idxs

