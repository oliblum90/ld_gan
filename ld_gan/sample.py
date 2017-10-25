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
                    sub_set_size = None,
                    same_class = False,
                    img_augmenter = None):
    
    while True:
        
        if sub_set_size is not None:
            x_temp_1 = X[np.random.randint(0, len(X), sub_set_size)]
        else:
            x_temp_1 = X
        
        if img_augmenter is not None:
            x = np.array([img_augmenter(img) for img in x_temp_1.copy()])
        else:
            x = x_temp_1
        
        log_time("get_z_enc")
        #z_enc = ld_gan.utils.model_handler.apply_model(enc, x, batch_size=batch_size)
        z_enc = ld_gan.utils.model_handler.apply_models(x, batch_size, enc)
        log_time("get_z_enc")
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        
        x_batch = x[batch_idxs]
        y_batch = y[batch_idxs]
        
        log_time("find_nn")
        
        # get surrounding
        sr_idxs = []
        if same_class:
            sorted_idxs = nn_gpu(z_enc, z_enc[batch_idxs], n_neighbors=None)
            for i in range(len(sorted_idxs)):
                iidxs = np.where(y[sorted_idxs[i]] == y_batch[i])[0]
                sr_idxs.append(sorted_idxs[i][iidxs][:nn_search_radius])
            sr_idxs = np.array(sr_idxs)
                
        else:
            sr_idxs = nn_gpu(z_enc, z_enc[batch_idxs], n_neighbors=nn_search_radius)
        
        # get nearest neighbors
        nn_idxs = [i[np.random.randint(0, len(i), n_neighbors)] for i in sr_idxs]
        
        log_time("find_nn")

        # get z_batch
        batch_z_all = z_enc[nn_idxs]
        rand_weights = np.random.rand(n_neighbors, batch_size)
        rand_weights = rand_weights / np.sum(rand_weights, axis=0)
        rand_weights = rand_weights.transpose()
        z_batch = [np.average(za, 0, w) for w, za in zip(rand_weights, batch_z_all)]
        z_batch = np.array(z_batch)
        
        z_batch_orig = z_enc[batch_idxs]
        
        yield x_batch, y_batch, z_batch, batch_idxs, nn_idxs, sr_idxs, z_enc, rand_weights

        
        
        
        
        
        
def nn_sampler_scs(enc, 
                   X, 
                   y,
                   batch_size, 
                   nn_search_radius = 50, 
                   n_neighbors = 5,
                   sub_set_size = None,
                   img_augmenter = None):
    
    latent_size = int(str(enc).split("Conv2d")[-1].split(", ")[1])
    
    while True:
        
        if sub_set_size is not None:
            x_temp_1 = X[np.random.randint(0, len(X), sub_set_size)]
        else:
            x_temp_1 = X
        
        if img_augmenter is not None:
            x = np.array([img_augmenter(img) for img in x_temp_1.copy()])
        else:
            x = x_temp_1
        
        batch_idxs = np.random.randint(0, len(x), batch_size)
        
        x_batch = x[batch_idxs]
        y_batch = y[batch_idxs]
        
        log_time("get_z_enc")
        z_enc = np.zeros((len(x), latent_size))
        i_y = np.concatenate([np.where(y==c)[0] for c in np.unique(y_batch)])
        z_enc[i_y] = ld_gan.utils.model_handler.apply_models(x[i_y], batch_size, enc)
        log_time("get_z_enc")
        
        log_time("find_nn")
        
        # get surrounding
        sr_idxs = []
        sorted_idxs = nn_gpu(z_enc, z_enc[batch_idxs], n_neighbors=None)
        for i in range(len(sorted_idxs)):
            iidxs = np.where(y[sorted_idxs[i]] == y_batch[i])[0]
            sr_idxs.append(sorted_idxs[i][iidxs][:nn_search_radius])
        sr_idxs = np.array(sr_idxs)

        # get nearest neighbors
        nn_idxs = [i[np.random.randint(0, len(i), n_neighbors)] for i in sr_idxs]
        
        log_time("find_nn")

        # get z_batch
        batch_z_all = z_enc[nn_idxs]
        rand_weights = np.random.rand(n_neighbors, batch_size)
        rand_weights = rand_weights / np.sum(rand_weights, axis=0)
        rand_weights = rand_weights.transpose()
        z_batch = [np.average(za, 0, w) for w, za in zip(rand_weights, batch_z_all)]
        z_batch = np.array(z_batch)
        
        z_batch_orig = z_enc[batch_idxs]
        
        yield x_batch, y_batch, z_batch, batch_idxs, nn_idxs, sr_idxs, z_enc, rand_weights

        
        
        
        
        
        

def same_class_sampler(enc, 
                       X, 
                       Y,
                       batch_size, 
                       n_neighbors = 5,
                       img_augmenter = None):
        
    while True:
        
        
        log_time("sample")
        
        n_classes = int(Y.max() + 1)
        batch_idxs = np.random.randint(0, len(X), batch_size)
        
        x_batch = X[batch_idxs]
        y_batch = Y[batch_idxs]
        
        all_idxs = np.arange(len(X))
        sr_idxs = [all_idxs[Y==y_batch[s]] for s in range(batch_size)]
        nn_idxs = [sr_idxs[s][np.random.randint(0, len(sr_idxs[s]), n_neighbors)] \
                   for s in range(batch_size)]
        x = X[nn_idxs]
        x = x.reshape(-1, X.shape[1], X.shape[2], X.shape[3])
        
        z_enc = ld_gan.utils.model_handler.apply_models(x, batch_size, enc)

        # get z_batch
        batch_z_all = z_enc.reshape(-1, n_neighbors, z_enc.shape[-1])
        rand_weights = np.random.rand(n_neighbors, batch_size)
        rand_weights = rand_weights / np.sum(rand_weights, axis=0)
        rand_weights = rand_weights.transpose()
        z_batch = [np.average(za, 0, w) for w, za in zip(rand_weights, batch_z_all)]
        z_batch = np.array(z_batch)
        
        log_time("sample")
        
        yield x_batch, y_batch, z_batch, batch_idxs, nn_idxs, sr_idxs, z_enc, rand_weights
        
        
def img_augmenter(img, max_zoom=0.8, lrflip=True, resize=64):
    
        img_size = int(np.random.uniform(max_zoom, 1.0) * img.shape[0])
        img_pos_x_min = np.random.randint(0, int((1. - max_zoom) * img_size))
        img_pos_x_max = img_pos_x_min + img_size
        img_pos_y_min = np.random.randint(0, int((1. - max_zoom) * img_size))
        img_pos_y_max = img_pos_y_min + img_size
        img = img[img_pos_x_min:img_pos_x_max, img_pos_y_min:img_pos_y_max]
        img = scipy.misc.imresize(img, (resize, resize))
        
        if 0.5 > np.random.rand():
            img = np.fliplr(img)
            
        return img
            
        