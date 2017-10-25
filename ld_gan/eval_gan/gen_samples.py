import os
import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import numpy as np
import sys
from tqdm import tqdm
import scipy.misc
import ld_gan.utils.utils as ld


def gen_samples(project,
                epoch,
                X, Y,
                Xt=None, Yt=None,
                path_dst = None,
                n_imgs_per_class = 1000,
                batch_size  = 256,
                nn_search_radius = 25,
                n_neighbors = 5,
                n_classes = None,
                create_nnmm=False):
    
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
    
    if n_classes is None:
        n_classes = Y.max() + 1

        
    samples = []
    nnmm_imgs = []
    nnmm_imgs_ext = []
    for c in tqdm(range(n_classes)):

        x = X[Y==c]
        y = Y[Y==c]

        sampler = ld_gan.sample.nn_sampler_life(enc, x, y, 
                                                batch_size, 
                                                nn_search_radius = nn_search_radius,
                                                n_neighbors = n_neighbors)

        n_iters = int(n_imgs_per_class / batch_size) + 1


        imgs = []
        nn_idxs = []
        for it in tqdm(range(n_iters)):
            _, _, Z, _, nn_i, _, _,w = sampler.next()
            nn_idxs.append(nn_i)
            imgs_gen = ld_gan.utils.model_handler.apply_model(gen, Z)
            imgs.append(imgs_gen)
        
        nn_idxs = np.concatenate(nn_idxs)
        imgs = np.concatenate(imgs)
        samples.append(imgs)

        if path_dst is not None:
            c_path = os.path.join(path_dst, "samples", str(c).zfill(3))
            ld.mkdir(c_path)
            fname = fname = os.path.join(c_path, "nn_idxs.txt")
            np.savetxt(fname, nn_idxs)
            fname = fname = os.path.join(c_path, "w.txt")
            np.savetxt(fname, w)

            for idx_img, img in enumerate(imgs):
                fname = os.path.join(c_path, str(idx_img).zfill(5) + ".jpg")
                scipy.misc.imsave(fname, img)
            
            if create_nnmm:
                
                xt = Xt[Yt==c]
                yt = Yt[Yt==c]
                
                c_path = os.path.join(path_dst, "nnmm", str(c).zfill(3))
                ld.mkdir(c_path)
                nnmm = ld_gan.eval_gan.missing_mode_nn.create_nn_imgs(x, xt, imgs, c_path)
                nnmm_ext = []
                for i in range(len(nnmm)):
                    temp = np.concatenate([nnmm[i], 
                                          np.ones((X.shape[1],int(X.shape[1]*0.3),3))*255, 
                                          np.concatenate(x[nn_idxs[i]], axis=1)], axis=1)
                    nnmm_ext.append(temp)
                    # SAVE IMGS
                nnmm_imgs.append(nnmm)
                nnmm_imgs_ext.append(nnmm_ext)
                
        
    samples = np.concatenate(samples)
        
    if create_nnmm:
        nnmm_imgs = np.concatenate(nnmm_imgs)
        nnmm_imgs_ext = np.concatenate(nnmm_imgs_ext)
        return samples, nnmm_imgs, nnmm_imgs_ext
    
    else:
        return samples

