import os
os.chdir("../")

import unittest

class TestSampler(unittest.TestCase):

    def test(self):
        
        import ld_gan
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        
        # load model
        project, epoch = "xf_11111_mc.py", 650
        enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        
        # load data
        X, Y = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)
        Y = np.argmax(Y, axis = 1)
        Z = ld_gan.utils.model_handler.apply_model(enc, X, batch_size=100)
        
        # create sampler and sample
        sampler = ld_gan.sample.nn_sampler_life(enc, X, Y, 256, n_neighbors=5)
        x, y, z, idx, idx_nn = sampler.next()
        
        # check that x/y/z correspond to idx/idx_nn
        x_np = ld_gan.data_proc.transform(x)
        y_np = ld_gan.data_proc.transform(y)
        z_np = ld_gan.data_proc.transform(z)
        idx_np = ld_gan.data_proc.transform(idx)
        idx_nn_np = ld_gan.data_proc.transform(idx_nn)
        
        # check that imgs are (almost) equal
        max_diff = np.abs(x_np.astype('float') - X[idx_np].astype('float')).max()
        self.assertLessEqual(max_diff, 1)
        
        # check that ys are equal
        self.assertEqual(np.all(y_np == Y[idx_np]), True)
        
        # check that nearest neighbors are nearly correct
        _, idxs_sknn = NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z[idx_np])
        
        # check that there is a linear combination to construct z
        
        # check that most indexes are used after two epoch
        

if __name__ == '__main__':
    unittest.main()