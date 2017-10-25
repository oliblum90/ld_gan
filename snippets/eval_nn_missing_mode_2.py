

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
        
        P_x  = "eval_imgs/nn_missing_mode/xS_111v1_gclf_scs_new.py/E30_sr25"
        P_X  = "data/faceScrub/crop_train/"
        P_Xt = "data/faceScrub/crop_test/"
        path_dst = "eval_imgs/nn_missing_mode/xS_111v1_nnscs_new.py/nnmm_E30_sr25"
        resize = 128
        
        classes = sorted(os.listdir(P_X))
        n_classes = len(classes)
        
        for c in range(n_classes):
        
            # load imgs
            path_x = os.path.join(P_x, classes[c])
            path_X = os.path.join(P_X, classes[c])
            path_Xt = os.path.join(P_Xt, classes[c])


            X, Y = ld_gan.data_proc.data_loader.load_data(path_X, 
                                                          verbose=0,
                                                          resize=resize, 
                                                          all_img_in_one_dir=True)
            Xt, Yt = ld_gan.data_proc.data_loader.load_data(path_Xt, 
                                                            verbose=0,
                                                            resize=resize, 
                                                            all_img_in_one_dir=True)
            x, y = ld_gan.data_proc.data_loader.load_data(path_x, 
                                                          verbose=2,
                                                          resize=resize, 
                                                          all_img_in_one_dir=True,
                                                          file_type=".jpg")

            # create nearest neighbor imgs
            path = os.path.join(path_dst, classes[c])
            ld.mkdir(path)
            ld_gan.eval_gan.missing_mode_nn.create_nn_imgs(X, Xt, x, path)
        