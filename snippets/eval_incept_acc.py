import os
os.chdir("../")


import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import ld_gan.utils.utils as ld
import numpy as np
import sys
from tqdm import tqdm
import scipy.misc
from sklearn.utils import shuffle


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)
        
        
        
        batch_size = 32
        
        
        
        
        # load model
        fname = "eval_imgs/clf_model/scrub/cnn_64_075.pth"
        cnn = ld_gan.utils.model_handler.load_model_with_different_gpu_id(fname)
        
        
        # load data
        path = "projects/XS_111v1_gclf_scs_new.py/incept_imgs/0010/"
        fnames = ld.listdir(path)
        X = [scipy.misc.imread(fname) for fname in tqdm(fnames)]
        X = [scipy.misc.imresize(img, (299, 299)) for img in tqdm(X)]
        Y = [int(fname.split("_")[-1].split(".")[0]) for fname in fnames]
        X = np.array(X)
        Y = np.array(Y)
        
        
        # get acc
        n_iters_test = int(len(X) / batch_size)
        acc = 0
        for it_test in range(n_iters_test):

            idx_lower = it_test * batch_size
            idx_upper = idx_lower + batch_size
                        
            x_batch = X[idx_lower : idx_upper]
            y_batch = Y[idx_lower : idx_upper]
                        
            x_batch = ld_gan.data_proc.transform(x_batch)
            yt_pred = cnn(x_batch)
            yt_pred = ld_gan.data_proc.transform(yt_pred)
            yt_pred = np.argmax(yt_pred, axis = 1)
            
            print "\n"
            print yt_pred
            print y_batch
            
            acc += float((yt_pred == y_batch).sum()) / len(y_batch)
            
        acc = acc / it_test
        print acc
        
        
        