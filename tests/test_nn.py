import os
os.chdir("../")

import unittest

class TestNN(unittest.TestCase):

    def test(self):
        
        import numpy as np
        from ld_gan.utils.nearest_neighbors import nn_gpu
        from sklearn.metrics.pairwise import pairwise_distances
        
        n_neighbors = 5
        
        # generate random points
        z_all = np.random.rand(8184, 256)
        z_batch = np.random.rand(256, 256)
        
        # compute nearest neighbors
        idxs = nn_gpu(z_batch, z_all, n_neighbors = n_neighbors)
        
        # compute nearest neighbors with sklearn
        dists = pairwise_distances(z_all, 
                                   z_batch, 
                                   metric='cosine')
        idxs_sk = np.argsort(dists, axis=1)[:, :n_neighbors]
        print idxs.shape
        print idxs_sk.shape

if __name__ == '__main__':
    unittest.main()