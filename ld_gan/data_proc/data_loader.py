
import os
import numpy as np
from tqdm import tqdm
import scipy.misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

PATH_MNIST   = "/export/home/oblum/projects/ls_gan/data/mnist/jpg_32"
PATH_FLOWER  = "/export/home/oblum/projects/ld_gan/data/flowers_102/jpg_256"
PATH_CELEBA  = "/export/home/oblum/projects/ld_gan/data/celebA/jpg_64"
PATH_CELEBA_128 = "/export/home/oblum/projects/ls_gan/data/celebA/jpg_128/faces"
PATH_FLOWER_17 = "/export/home/oblum/projects/ls_gan/data/flowers_17/jpg_64"
PATH_BIRDS = "/export/home/oblum/projects/ld_gan/data/birds/images"


def load_data(path, 
              index_list = None,
              random_state=42, 
              verbose = 1, 
              n_jobs = 1,
              resize = None):
    
    if path == -1:
        path = PATH_FLOWER_17
    if path == 0:
        path = PATH_MNIST
    elif path == 1:
        path = PATH_FLOWER
    elif path == 2:
        path = PATH_CELEBA
    elif path == 3:
        path = PATH_BIRDS
        
    print "load data from '{}'".format(path)
    
    class_dirs = [os.path.join(path, c) for c in os.listdir(path)]
    n_classes = len(class_dirs) + 1
    
    X = []
    y = []
    
    iterator = range(len(class_dirs))
    
    if verbose >= 1:
        iterator = tqdm(iterator)
    
    for idx in iterator:
        
        c = sorted(class_dirs)[idx]
        
        if index_list is None:
            fnames = [os.path.join(c, f) for f in sorted(os.listdir(c))]
        else:
            def name2idx(fname):
                return int(fname[-9:-4])
            fnames = [os.path.join(c, f) for f in sorted(os.listdir(c)) \
                                         if name2idx(f) in index_list]
        
        iterator_2 = sorted(fnames)
        if verbose == 2:
            iterator_2 = tqdm(fnames)
                
        if n_jobs > 1:
            X_c = _imap_unordered_bar(scipy.misc.imread, fnames, n_jobs)
        else:
            X_c = [scipy.misc.imread(f) for f in iterator_2]
        y_c = [idx] * len(X_c)
        
        X = X + X_c
        y = y + y_c
        
    if resize is not None:
        X = [scipy.misc.imresize(img, (resize, resize)) for img in X]
        
    X = np.array(X)
    y = np.array(y)
    
    X, y = shuffle(X, y, random_state = random_state)
    
    y = np.eye(n_classes - 1)[y]
    
    return X, y
    
    

    
def load_data_split(path, 
                    random_state=42, 
                    verbose = 0, 
                    n_jobs = 1,
                    resize = None):
    pass
    
    
