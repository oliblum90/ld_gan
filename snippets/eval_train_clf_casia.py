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
from sklearn.utils import shuffle
import ld_gan.utils.utils as ld


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)


        path = "/net/hci-storage02/groupfolders/compvis/datasets/CASIA-WebFace/CASIA-WebFace"
        
        classes = {}
        for c, c_name in enumerate(sorted(os.listdir(path))):
            classes[c_name] = c
        
        fnames = ld.listdir(path, ".jpg", True)
        Y = np.array([classes[fname.split("/")[-2]] for fname in fnames])
        
        
        n_classes = len(classes)
        
        ld_gan.eval_gan.clf.train_cnn_load_live(fnames, Y, n_classes,
                                                64, 0.001, 500, 5, 
                                                "eval_imgs/clf_model/casia/")
        