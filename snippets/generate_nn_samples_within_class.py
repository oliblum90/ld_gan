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
import ld_gan.utils.utils as ld



if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)

        SAVE_PATH = "eval_imgs/xs_111v1.py_NN_CLASS" # HERE!!!
        N_IMG_PER_CLASS = 100 # HERE!!!
        BATCH_SIZE  = 100
        PROJECT = "xs_111v1.py" # HERE!!!
        EPOCH = 50 # HERE!!
        nn_search_radius = 25
        n_neighbors = 5

        
        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")
        
        
        
        X, Y = ld_gan.data_proc.data_loader.load_data(52, resize = 64) # HERE!!!
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]

        
        
        for c in tqdm(range(n_classes)):
                        
            x = X[Y==c]
            y = Y[Y==c]
                
            sampler = ld_gan.sample.nn_sampler_life(enc, x, y, 
                                                    BATCH_SIZE, 
                                                    nn_search_radius = nn_search_radius,
                                                    n_neighbors = n_neighbors)

            n_iters = int(N_IMG_PER_CLASS / BATCH_SIZE) + 1 

            
            save_path = os.path.join(SAVE_PATH, str(c).zfill(3))
            ld.mkdir(save_path)
            
            imgs = []
            for it in tqdm(range(n_iters)):

                _, _, Z, _, nn_idxs, _, _ = sampler.next()
                imgs.append(ld_gan.utils.model_handler.apply_model(gen, Z))
            imgs = np.concatenate(imgs)
            
            c_path = os.path.join(SAVE_PATH, str(c).zfill(3))
            for idx_img, img in enumerate(imgs):
                fname = os.path.join(c_path, str(idx_img).zfill(5) + ".jpg")
                scipy.misc.imsave(fname, img)



