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


        path_train = "data/faceScrub/imgs_top_aligned_train/"
        path_test = "data/faceScrub/imgs_top_aligned_test/"
        save_model_path = "eval_imgs/clf_model/scrub_top_aligned_3/"
        
        ld_gan.eval_gan.clf.train_cnn_load_live(path_train, path_test,
                                                batch_size = 64, 
                                                lr = 0.001, 
                                                n_epochs = 1000, 
                                                save_model_step = 25, 
                                                save_model_path = save_model_path,
                                                resize = 299)