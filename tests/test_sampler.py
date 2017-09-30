import unittest

class TestSampler(unittest.TestCase):

    def test_nn_sampler_life(self):
        
        import ld_gan
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from ld_gan.utils.nearest_neighbors import nn_gpu
        
        # load model
        project, epoch = "xf_11111_t1.py", 650
        enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        
        # load data
        X, Y = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)
        Y = np.argmax(Y, axis = 1)
        Z = ld_gan.utils.model_handler.apply_model(enc, X, batch_size=100)
        
        # create sampler and sample
        sampler = ld_gan.sample.nn_sampler_life(enc, X, Y, 256, 
                                                n_neighbors=5,
                                                nn_search_radius = 50)
        x, y, z, z_orig, batch_idxs, idxs = sampler.next()
                
        # check that x/y/z correspond to idx/idx_nn
        self.assertEqual(np.all(x == X[batch_idxs]), True)
        self.assertEqual(np.all(y == Y[batch_idxs]), True)
        self.assertEqual(np.all(z_orig == Z[batch_idxs]), True)
        
        # check nearest neighbors
        nn_50 = nn_gpu(Z, z_orig, n_neighbors=50)
        for i, n in zip(idxs, nn_50):
            for ii in i:
                self.assertEqual(ii in n, True)
        
        # check that z is a linear combination of its neighbors
        for z_nn_single, z_single in zip(Z[idxs], z):
            z_nn_matrix = np.concatenate([z_nn_single, np.array([z_single])])
            rank = np.linalg.matrix_rank(z_nn_matrix)
            self.assertLessEqual(rank, 5)
        
if __name__ == "__main__":
    unittest.main()
    