

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
        
        
        project      = "XS_111v1_gclf_scs_new.py"
        epoch        = 10
        resize       = 128
        n_imgs       = 10000
        batch_size   = 1024
        nn_sr        = 50
        n_neighbors  = 5
        n_classes    = 25
        
        
        gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
        enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        
        
        path_dst = "eval_imgs/nn_missing_mode/" + project
        
        
        path = "data/faceScrub/imgs_top_aligned/"
        X, Y = ld_gan.data_proc.data_loader.load_data(path, 
                                                      resize = resize, 
                                                      test_train="train")
        Y = np.argmax(Y, axis = 1)

        path = "data/faceScrub/imgs_top_aligned/"
        Xt, Yt = ld_gan.data_proc.data_loader.load_data(path, 
                                                        resize=resize, 
                                                        test_train="test")
        Yt = np.argmax(Yt, axis = 1)




        samples,nnmm,nnmm_ext=ld_gan.eval_gan.gen_samples(project, 
                                                          epoch,
                                                          X, Y, Xt, Yt,
                                                          path_dst = path_dst,
                                                          n_imgs_per_class=n_imgs,
                                                          batch_size  = batch_size,
                                                          nn_search_radius = nn_sr,
                                                          n_neighbors = n_neighbors,
                                                          n_classes = n_classes,
                                                          create_nnmm = True)

