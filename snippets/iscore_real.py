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


        PATH = "eval_imgs"
        BATCH_SIZE = 32


        imgs_real, _ = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)
        i_score = ld_gan.eval_gan.InceptionScore()
        score = i_score.score(imgs_real, batch_size=BATCH_SIZE)

        print "finished"
        print "----------------------------------------"
        print "score = ", score

        np.savetxt(os.path.join(PATH, "i_score_flowers_real.txt"), np.array([score]))