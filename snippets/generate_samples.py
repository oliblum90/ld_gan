import os
os.chdir("../")

import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import numpy as np
import sys
from tqdm import tqdm


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)

        N_IMG = 50000
        BATCH_SIZE  = 256
        PROJECT = "xf_111v1.py"
        EPOCH = 350


        X, Y = ld_gan.data_proc.data_loader.load_data(2, verbose=2)
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]


        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")

        sampler = ld_gan.sample.nn_sampler_life(enc, X, Y, 
                                                BATCH_SIZE, 
                                                nn_search_radius = 10,
                                                n_neighbors = 2)
        
        n_iters = int(50000 / BATCH_SIZE) + 1 
        
        for i in tqdm(range(n_iters)):
            
            X, Y, Z, _, _, _, _ = sampler.next()

        

