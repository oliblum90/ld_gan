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

        SAVE_PATH = "eval_imgs/xs_111v1.py" # HERE!!!
        N_IMG_PER_CLASS = 1000 # HERE!!!
        BATCH_SIZE  = 256
        PROJECT = "xf_111v1_split.py" # HERE!!!
        EPOCH = 350
        N_INTERPOL_IMGS = 5
        
        
        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")
        dis = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "dis")

        n_iters = int(N_IMG_PER_CLASS / BATCH_SIZE) + 1 
        
        
        for c in range(102):
            
            print "CLASS:", c

            path = "data/flowers_102/jpg_train_256/" + str(c+1)
            imgs = np.array([scipy.misc.imread(n) for n in ld.listdir(path)])
            imgs = np.array([scipy.misc.imresize(i, (64,64)) for i in imgs])

            for it in tqdm(range(n_iters)):
                
                
                idxs = np.random.randint(0, len(imgs), BATCH_SIZE*N_INTERPOL_IMGS)
                z = ld_gan.utils.model_handler.apply_model(enc, imgs[idxs], BATCH_SIZE)
                z = z.reshape((BATCH_SIZE, N_INTERPOL_IMGS, -1))
                
                
                #z = np.mean(z, axis=1)
                
                
                rand_weights = np.random.rand(N_INTERPOL_IMGS, BATCH_SIZE)
                rand_weights = rand_weights / np.sum(rand_weights, axis=0)
                rand_weights = rand_weights.transpose()
                z = [np.average(za, 0, w) for w, za in zip(rand_weights, z)]
                z = np.array(z)
                
                
                
                x = ld_gan.utils.model_handler.apply_model(gen, z, BATCH_SIZE)
                d = ld_gan.utils.model_handler.apply_model(dis, x, BATCH_SIZE)
                
                for idx, img in enumerate(x):
                    idx_img = str(it * BATCH_SIZE + idx).zfill(5)
                    class_img = str(c).zfill(3)
                    dis_score = str(d[idx])
                    idxs_img = str(idxs[idx])
                    save_path = os.path.join(SAVE_PATH, class_img)
                    ld.mkdir(save_path)
                    fname = os.path.join(save_path, class_img + "_" + \
                                                    idx_img  + "_" + \
                                                    dis_score + "_" + \
                                                    idxs_img + \
                                                    '.png')
                    scipy.misc.imsave(fname, img)

        

