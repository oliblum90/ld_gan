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


        path = "data/faceScrub/imgs_top_aligned/"
        X, Y = ld_gan.data_proc.data_loader.load_data(path, resize=128, test_train="train")
        Xt, Yt = ld_gan.data_proc.data_loader.load_data(path, resize=128, test_train="test")
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        Yt = np.argmax(Yt, axis = 1)
        img_size = X.shape[2]
        X = np.array([scipy.misc.imresize(x, (299, 299)) for x in X])
        Xt = np.array([scipy.misc.imresize(x, (299, 299)) for x in Xt])
        
        ld_gan.eval_gan.clf.train_cnn(X, Y, Xt, Yt, n_classes,
                                      64, 0.001, 1000, 25, 
                                      "eval_imgs/clf_model/scrub_top_aligned_2/")