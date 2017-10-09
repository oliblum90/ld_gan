import os
os.chdir("../")

import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import numpy as np
import sys
from tqdm import tqdm
import scipy.misc
import ld_gan
from ld_gan.utils.nearest_neighbors import nn_gpu


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)

        SAVE_PATH = "eval_imgs/all_classes/xf_111v1.py"
        N_IMGS_PER_CLASS = 10
        PROJECT = "xf_111v1.py"
        EPOCH = 350
        N_INTERPOLS = 7

        X, Y = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]


        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")


        Z = ld_gan.utils.model_handler.apply_model(enc, X, 100)
        
        
        for c in tqdm(range(n_classes)):
            
            xc = X[Y==c]
            zc = Z[Y==c]
            
            x_batch = xc[np.random.randint(0, len(xc), N_IMGS_PER_CLASS)]
            z_batch = ld_gan.utils.model_handler.apply_model(enc, x_batch)
            
            nn_idxs = nn_gpu(zc, z_batch, 2)
            
            x_pairs = xc[nn_idxs]
            
            for idx, x_pair in enumerate(x_pairs):
                
                img1, img2 = x_pair
                img = ld_gan.utils.model_handler.get_interpol_imgs(enc, gen, 
                                                                   img1, img2, 
                                                                   n_interpols=N_INTERPOLS)
                
                fname = os.path.join(SAVE_PATH, str(c).zfill(2) + "_" + \
                                                str(idx).zfill(3) + ".png")
                scipy.misc.imsave(fname, img)
            
            
            
            