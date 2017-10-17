

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
        
        for c in range(102):
        
            # load imgs
            path_x = "eval_imgs/xf_111v1_split.py_CLASS_RANDWEIGHT/" + str(c).zfill(3)
            path_X = "data/flowers_102/jpg_train_256/" + str(c+1)
            path_Xt = "data/flowers_102/jpg_test_256/" + str(c+1)


            X, Y = ld_gan.data_proc.data_loader.load_data(path_X, 
                                                          verbose=0,
                                                          resize=64, 
                                                          all_img_in_one_dir=True)
            Xt, Yt = ld_gan.data_proc.data_loader.load_data(path_Xt, 
                                                            verbose=0,
                                                            resize=64, 
                                                            all_img_in_one_dir=True)
            x, y = ld_gan.data_proc.data_loader.load_data(path_x, 
                                                          verbose=2,
                                                          resize=64, 
                                                          all_img_in_one_dir=True)

            # create nearest neighbor imgs
            path = "eval_imgs/nn_missing_mode/flowers_class_randweight/" + str(c)
            ld.mkdir(path)
            ld_gan.eval_gan.missing_mode_nn.create_nn_imgs(X, Xt, x, path)
        