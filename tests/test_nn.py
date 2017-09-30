import unittest

class TestNN(unittest.TestCase):

    def test(self):
        
        import numpy as np
        from ld_gan.utils.nearest_neighbors import nn_gpu
        from sklearn.metrics.pairwise import pairwise_distances
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = 5
        
        # generate random points
        z_all = np.random.rand(8184, 256)
        z_batch = np.random.rand(256, 256)
        
        # compute nearest neighbors
        idxs = nn_gpu(z_all, z_batch, n_neighbors = n_neighbors)
        
        # compute nearest neighbors with sklearn pairwise distance
        dists = pairwise_distances(z_batch,
                                   z_all, 
                                   metric='cosine')
        idxs_sk = np.argsort(dists, axis=1)[:, :n_neighbors]
        
        # compute nearest neighbors with sklearn NearestNeighbors
        nn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors).fit(z_all)
        _, idxs_sk_nn = nn.kneighbors(z_batch)
                
        self.assertEqual(np.all(idxs == idxs_sk), True)
        self.assertEqual(np.all(idxs == idxs_sk_nn), True)

if __name__ == "__main__":
    unittest.main()