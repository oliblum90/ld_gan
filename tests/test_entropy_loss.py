import unittest

class TestSampler(unittest.TestCase):

    def test_nn_sampler_life(self):
        
        import ld_gan
        import numpy as np
        from ld_gan.train_ops.entropy_enc import entropy_loss
        import torch
        
        # set parameters
        batch_size = 256
        
        # load models
        project, epoch = "xf_11111.py", 400
        enc1 = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        project, epoch = "xf_11111_kde10.py", 400
        enc2 = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        
        # load data
        X, Y = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)
        Y = np.argmax(Y, axis = 1)
        Z1 = ld_gan.utils.model_handler.apply_model(enc1, X, batch_size=100)
        Z2 = ld_gan.utils.model_handler.apply_model(enc2, X, batch_size=100)
        
        batch_idxs = np.random.randint(0, len(Y), batch_size)
        z1 = Z1[batch_idxs]
        z2 = Z2[batch_idxs]
        
        Z1, Z2, z1, z2 = ld_gan.data_proc.transform(Z1, Z2, z1, z2)
        
        Z1 = torch.squeeze(Z1)
        Z2 = torch.squeeze(Z1)
        z1 = torch.squeeze(z1)
        z2 = torch.squeeze(z2)
        
        l1 = entropy_loss(Z1, z1)
        l2 = entropy_loss(Z2, z2)
        d = l1 - l2
        d = d.data.cpu().numpy()[0]

        self.assertLess(0, d)
        
        
if __name__ == "__main__":
    unittest.main()