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
from multiprocessing import Process, Queue

def precomputing_iterator(iterator, maxsize = 5):
    
    def enqueue(q):
        while True:
            q.put(iterator.next())
    
    q = Queue(maxsize = maxsize)
    p = Process(target=enqueue, args=(q,))
    p.start()
    
    while True:
        yield q.get()

#0
def generate_rand_noise(imgs, y, batch_size, latent_size):
    
    while True:
        z_batch = np.random.uniform(-1, 1, (int(batch_size), 
                                            int(latent_size)))
        
        batch_idxs = np.random.randint(0, len(imgs), batch_size)
        img_batch = imgs[batch_idxs]
        y_batch = y[batch_idxs]
        
        yield z_batch, z_batch, img_batch, y_batch

#1
def pick_rand_sample(z_enc, imgs, y, batch_size):
    
    while True:
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        z_batch = z_enc[batch_idxs]
        img_batch = imgs[batch_idxs]
        y_batch = y[batch_idxs]
        yield z_batch, z_batch, img_batch, y_batch
        
#2 
def pick_rand_sample_with_weighted_noise(z_enc, batch_size, noise_amount = 1):
    
    while True:
        noise = np.random.rand(batch_size, z_enc.shape[1])
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        z_batch = z_enc[batch_idxs]
        z_batch_noise = z_batch + (noise * z_batch * noise_amount)
        # yield z_batch, batch_idxs
        yield z_batch, z_batch, None

#3 : uniform
        
#4
def pick_rand_sample_with_noise(z_enc, batch_size, noise_amount = 0.05):
    
    while True:
        noise = np.random.uniform(-1, 1, (batch_size, z_enc.shape[1]))        
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        z_batch = z_enc[batch_idxs]
        
        z_batch_noise = (1. - noise_amount) * z_batch + noise * noise_amount
        
        # yield z_batch, batch_idxs
        yield z_batch_noise, z_batch_noise, None
        
#5
def pick_rand_sample_with_concat_noise(z_enc, imgs, y, batch_size, latent_size = 300):
    
    while True:
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        z_batch = z_enc[batch_idxs]
        
        n_size = latent_size - z_batch.shape[1]
        noise = np.random.uniform(-1, 1, (batch_size, n_size))

        z_batch_con = np.concatenate((z_batch, noise), axis=1)
        
        img_batch = imgs[batch_idxs]
        y_batch = y[batch_idxs]
        
        yield z_batch_con, z_batch, img_batch, y_batch

        
#6      
def sample_with_kde(z_enc, imgs, y, batch_size, bandwidth = None, n_jobs = 10):
    
    if bandwidth is None:
        bandwidth = find_ideal_kde_sampling_bandwidth(z_enc)
        print "found ideal bandwidth {}".format(bandwidth)
    
    kde = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(z_enc)
    nbrs = NearestNeighbors(n_neighbors = 1, n_jobs = n_jobs).fit(z_enc)
    
    while True:
        
        z_sampled = kde.sample(batch_size)
        
        dists, idxs = nbrs.kneighbors(z_sampled)
        idxs = idxs[:, 0]
        img_batch = imgs[idxs]
        y_batch = y[idxs]
        
        yield z_sampled, z_sampled, img_batch, y_batch
        
        
def sample_with_kde_rand_img(z_enc, imgs, y, batch_size, bandwidth = None, n_jobs = 10):
    
    if bandwidth is None:
        bandwidth = find_ideal_kde_sampling_bandwidth(z_enc)
        print "found ideal bandwidth {}".format(bandwidth)
    
    kde = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(z_enc)
    nbrs = NearestNeighbors(n_neighbors = 1, n_jobs = n_jobs).fit(z_enc)
    
    while True:
        
        z_sampled = kde.sample(batch_size)
        
        idxs = np.random.randint(0, len(z_enc), batch_size)
        img_batch = imgs[idxs]
        y_batch = y[idxs]
        
        yield z_sampled, z_sampled, img_batch, y_batch
        
        
#7   
def sample_kde_with_concat_noise(z_enc, 
                                 imgs, 
                                 batch_size, 
                                 latent_size = 300, 
                                 bandwidth = None,
                                 n_jobs = 10):
    
    if bandwidth is None:
        bandwidth = find_ideal_kde_sampling_bandwidth(z_enc)
        print "found ideal bandwidth {}".format(bandwidth)
        
    kde = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(z_enc)
    nbrs = NearestNeighbors(n_neighbors = 1, n_jobs = n_jobs).fit(z_enc)
    
    while True:
        
        n_size = latent_size - z_enc.shape[1]
        noise = np.random.uniform(-1, 1, (batch_size, n_size))
        
        z_sampled = kde.sample(batch_size)
        z_sampled_noise = np.concatenate((z_sampled, noise), axis=1)
        
        dists, idxs = nbrs.kneighbors(z_sampled)
        idxs = idxs[:, 0]
        img_batch = imgs[idxs]
        
        yield z_sampled_noise, z_sampled, img_batch
        
        
def sample_kde_with_concat_noise_rand_img(z_enc, 
                                          imgs, 
                                          batch_size, 
                                          latent_size = 300, 
                                          bandwidth = None,
                                          n_jobs = 10):
    
    if bandwidth is None:
        bandwidth = find_ideal_kde_sampling_bandwidth(z_enc)
        print "found ideal bandwidth {}".format(bandwidth)
    
    kde = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(z_enc)
    
    while True:
        
        n_size = latent_size - z_enc.shape[1]
        noise = np.random.uniform(-1, 1, (batch_size, n_size))
        
        z_sampled = kde.sample(batch_size)
        z_sampled_noise = np.concatenate((z_sampled, noise), axis=1)
        
        idxs = np.random.randint(0, len(z_enc), batch_size)
        img_batch = imgs[idxs]
        
        yield z_sampled_noise, z_sampled, img_batch


#8
def pick_rand_sample_dropout_noise(z_enc, imgs, batch_size, dropout = 0.5):
    
    latent_size = z_enc.shape[1]
    
    while True:
        
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        
        z_batch = z_enc[batch_idxs]
        img_batch = imgs[batch_idxs]
        
        z_batch_noise = np.copy(z_batch)
        n_do = int(latent_size * dropout)
        for b in range(batch_size):
            do_idxs = np.random.choice(np.arange(latent_size), n_do, False)
            z_batch_noise[b, do_idxs] = np.random.uniform(-1, 1, (n_do))
        
        yield z_batch_noise, z_batch, img_batch


#9
def sample_tsne_features(z_enc,
                         batch_size,
                         latent_size = 100,
                         n_neighbors = 5,
                         sampling_bw = 0.2,
                         fname_z_enc = "data/z_enc/z_enc.npy",
                         fname_tsne_pts = "data/z_enc/nn_tsne_mapped.npy",
                         norm = True):
    
    pts_maped = np.load(fname_tsne_pts)
    
    if norm:
        pts_maped[:,0] = pts_maped[:,0] / np.max(np.abs(pts_maped[:,0]))
        pts_maped[:,1] = pts_maped[:,1] / np.max(np.abs(pts_maped[:,1]))
        
    
    mapping_dim = pts_maped.shape[1]
    
    kde = KernelDensity(kernel='gaussian', bandwidth = sampling_bw).fit(pts_maped)
    nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(pts_maped)
    
    noise_size = latent_size - mapping_dim
    
    while True:
        
        noise = np.random.uniform(-1, 1, (batch_size, noise_size))
        
        z_ts = kde.sample(batch_size)
        dists, idxs = nbrs.kneighbors(z_ts)
        z_fs = z_enc[idxs]
                
        z_in  = np.concatenate((z_ts, noise), axis=1)
        z_out = z_enc[idxs].mean(axis = 1)
        
        yield z_in, z_out, None
        



def find_ideal_kde_sampling_bandwidth(ipt, 
                                      project = None, 
                                      enc_epoch = None, 
                                      n_pts = 2000):
    
    if ipt is None:
        
        from keras.models import load_model
        import tensorflow as tf

        enc = load_model("projects/" + project + "/model/enc.h5", 
                         custom_objects={"tf": tf})
        if enc_epoch is not None:
            enc_epoch_str = str(enc_epoch).zfill(4)
            enc.load_weights("projects/" + project + "/model/e_" + enc_epoch_str + ".h5")
        else:
            enc.load_weights("projects/" + project + "/model/enc_w_0.h5")

        ipt = enc.predict(ipt[:n_pts])
        
    z_enc = ipt[:n_pts]
    
    # get mean distance between samples
    nbrs = NearestNeighbors(n_neighbors = 2, n_jobs = 10).fit(z_enc)
    dists, _ = nbrs.kneighbors(z_enc)
    dist_mean =  dists[:,1].mean()
    
    # get mean distance from sample to nearest neighbor in real data
    kde = KernelDensity(kernel='gaussian', bandwidth = 1.0).fit(z_enc)
    z_sampled = kde.sample(n_pts)
    nbrs = NearestNeighbors(n_neighbors = 1, n_jobs = 10).fit(z_sampled)
    dists, _ = nbrs.kneighbors(z_enc)
    dist_mean_sampled =  dists.mean()
    
    bandwidth_ideal = dist_mean / dist_mean_sampled
    
    return bandwidth_ideal
    
    
    