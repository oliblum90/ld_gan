import os
import ld_gan
from time import time
import numpy as np
from ld_gan.utils.model_handler import apply_model
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import scipy.misc


PATH  = "/export/home/oblum/projects/ls_gan/data/flowers_102/jpg_128"
PATH_VGG  = "/export/home/oblum/projects/ld_gan/data/flowers_102/jpg_224"
PATH_FEATURES = "/export/home/oblum/projects/ld_gan/data/vgg_features.npy"

def load_imgs_and_vgg(path = PATH,
                      path_vgg = PATH_VGG, 
                      path_features = PATH_FEATURES,
                      resize = 64):
    
    try:
        features = np.load(path_features)
    
    except:
        paths = [os.path.join(path_vgg, f) for f in sorted(os.listdir(path_vgg))]
        fnames = []
        for p in paths:
            fnames = fnames + [os.path.join(p, f) for f in sorted(os.listdir(p))]
        imgs = [scipy.misc.imread(f) for f in tqdm(fnames)]
        imgs = np.array(imgs)
        
        model = models.vgg19_bn(pretrained=True)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        model.cuda()
        model.eval()
        
        features = ld_gan.utils.model_handler.apply_model(model, imgs, 16)
        
        np.save(path_features, features)
        
    paths = [os.path.join(path, f) for f in sorted(os.listdir(path))]
    fnames = []
    y = []
    for idx, p in enumerate(paths):
        fnames = fnames + [os.path.join(p, f) for f in sorted(os.listdir(p))]
        y.append(np.ones(len(os.listdir(p)))* idx)
    print fnames[2042]
    imgs = [scipy.misc.imread(f) for f in tqdm(fnames)]
    imgs = [scipy.misc.imresize(img, (resize, resize)) for img in imgs]
    imgs = np.array(imgs)
    y = np.concatenate(y)
    
    return imgs, y, features
    
    
    