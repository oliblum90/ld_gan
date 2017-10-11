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


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)

        SAVE_PATH = "eval_imgs/xf_111v1_split.py_TEST" # HERE!!!
        N_IMG = 5000 # HERE!!!
        BATCH_SIZE  = 512
        PROJECT = "xf_111v1_split.py" # HERE!!!
        EPOCH = 350

        X, Y = ld_gan.data_proc.data_loader.load_data(11, resize = 64) # HERE!!!
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]


        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")

        sampler = ld_gan.sample.nn_sampler_life(enc, X, Y, 
                                                BATCH_SIZE, 
                                                nn_search_radius = 20,
                                                n_neighbors = 5)
        
        n_iters = int(N_IMG / BATCH_SIZE) + 1 
        
        for it in tqdm(range(n_iters)):
            
            _, _, Z, _, nn_idxs, _, _ = sampler.next()
            
            # get class
            nn_idxs = np.array(nn_idxs)
            y = Y[nn_idxs[:, 0]] == Y[nn_idxs[:, 1]]
            y = y.astype(np.int)
            y[y==0] = -1
            y[y==1] = Y[nn_idxs[:, 0]][y==1]
            
            imgs = ld_gan.utils.model_handler.apply_model(gen, Z)
            for idx, img in enumerate(imgs):
                idx_img = str(it * BATCH_SIZE + idx)
                class_img = str(y[idx])
                fname = os.path.join(SAVE_PATH, idx_img + "_" + class_img + '.png')
                scipy.misc.imsave(fname, img)

        

