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

        SAVE_PATH = "eval_imgs/iscores/xs_111v1_nnscs_trip_f5/E30_sr_25"
        N_IMGS = 50000
        BATCH_SIZE  = 256
        PROJECT = "xs_111v1_gclf_scs_new.py"
        EPOCH = 50
        nn_search_radius = 25
        n_neighbors = 5
        data_path = "data/faceScrub/crop_train/"
        resize = 64
        model_fname = "eval_imgs/clf_model/scrub/cnn_64_075.pth"

        ld.mkdir(SAVE_PATH)
        
        # load data / model / sampler
        gen = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "gen")
        enc = ld_gan.utils.model_handler.load_model(PROJECT, EPOCH, "enc")
        
        X, Y = ld_gan.data_proc.data_loader.load_data(data_path, resize = resize)
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]
                
        sampler = ld_gan.sample.nn_sampler_life(enc, X, Y, 
                                                BATCH_SIZE, 
                                                nn_search_radius = nn_search_radius,
                                                n_neighbors = n_neighbors,
                                                same_class = True)

        
        # generate imgs
        n_iters = int(N_IMGS / BATCH_SIZE) + 1                        
        imgs = []
        ys = []
        for it in tqdm(range(n_iters)):
            _, Y, Z, _, _, _, _ = sampler.next()
            imgs.append(ld_gan.utils.model_handler.apply_model(gen, Z))
            ys.append(Y)
        imgs = np.concatenate(imgs)
        ys = np.concatenate(ys)

        
        # save imgs
        for idx in range(len(imgs)):
            fname = str(idx).zfill(6) + "_" + str(ys[idx]).zfill(3) + ".jpg"
            fname = os.path.join(SAVE_PATH, fname)
            scipy.misc.imsave(fname, imgs[idx])
            
        print "compute inception score..."
        model = ld_gan.utils.model_handler.load_model_with_different_gpu_id(model_fname)
        model.eval()
        i_score = ld_gan.eval_gan.InceptionScore(model=model)
        score = i_score.score(imgs, batch_size=16)
        print "score: {}".format(score)
        fname = os.path.join(SAVE_PATH, "iscore.txt")
        np.savetxt(fname, np.array([score]))
        