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

        SAVE_PATH = "eval_imgs/train_augment/XS_111v1_gclf_scs_new"
        #DATA_PATH = "data/faceScrub/crop_train/"
        DATA_PATH = "data/faceScrub/imgs_top_aligned/"
        N_IMG_PER_CLASS = 200 # HERE!!!
        BATCH_SIZE  = 80
        PROJECT = "XS_111v1_gclf_scs_new.py" # HERE!!!
        EPOCH = 10 # HERE!!
        resize = 128
        nn_search_radius = 50
        n_neighbors = 5

        
        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")
        
        # test
        t1 = ld_gan.utils.model_handler.apply_model(enc, np.zeros((BATCH_SIZE,128,128,3)))
        t2 = ld_gan.utils.model_handler.apply_model(gen, np.zeros((BATCH_SIZE, 512)))
        
        
        
        X, Y = ld_gan.data_proc.data_loader.load_data(DATA_PATH, 
                                                      resize=resize, 
                                                      test_train="train") 
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]

        classes = sorted(os.listdir(DATA_PATH))
        
        for c in tqdm(range(n_classes)):
                        
            x = X[Y==c]
            y = Y[Y==c]
                
            sampler = ld_gan.sample.nn_sampler_life(enc, x, y, 
                                                    BATCH_SIZE, 
                                                    nn_search_radius = nn_search_radius,
                                                    n_neighbors = n_neighbors)

            n_iters = int(N_IMG_PER_CLASS / BATCH_SIZE) + 1 

            
            save_path = os.path.join(SAVE_PATH, classes[c])
            ld.mkdir(save_path)
            
            imgs = []
            for it in tqdm(range(n_iters)):

                _, _, Z, _, nn_idxs, _, _, _ = sampler.next()
                imgs.append(ld_gan.utils.model_handler.apply_model(gen, Z))
            imgs = np.concatenate(imgs)
            
            c_path = os.path.join(SAVE_PATH, classes[c])
            for idx_img, img in enumerate(imgs):
                fname = os.path.join(c_path, str(idx_img).zfill(5) + ".jpg")
                scipy.misc.imsave(fname, img)



