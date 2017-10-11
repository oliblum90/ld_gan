

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
        
        # load imgs
        path = "eval_imgs/xf_111v1_split.py/" # HERE
        fnames = [os.path.join(path, fname) for fname in tqdm(os.listdir(path))]
        x = [scipy.misc.imread(fname) for fname in tqdm(fnames)]
        x = np.array(x)
        
        X, Y = ld_gan.data_proc.data_loader.load_data(10, resize=64) # HERE
        
        Xt, Yt = ld_gan.data_proc.data_loader.load_data(11, resize=64) # HERE
        Xt = Xt[:250]
                
        # create nearest neighbor imgs
        path = "eval_imgs/nn_missing_mode/flowers/"
        ld_gan.eval_gan.missing_mode_nn.create_nn_imgs(X, Xt, x, path)
        