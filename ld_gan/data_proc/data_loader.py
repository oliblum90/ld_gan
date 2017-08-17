
import os
import numpy as np
from tqdm import tqdm
import scipy.misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

PATH_MNIST   = "/export/home/oblum/projects/ls_gan/data/mnist/jpg_32"
PATH_FLOWER  = "/export/home/oblum/projects/ls_gan/data/flowers_102/jpg_128"
PATH_CELEBA  = "/export/home/oblum/projects/ls_gan/data/celebA/jpg_128"

def load_data(path, 
              split_test_train_ratio = None, 
              random_state=42, 
              verbose = 0, 
              n_jobs = 1,
              resize = None):
    
    if path == 0:
        path = PATH_MNIST
    elif path == 1:
        path = PATH_FLOWER
    elif path == 2:
        path = PATH_CELEBA
    
    print "load data from '{}'".format(path)
    
    class_dirs = [os.path.join(path, c) for c in os.listdir(path)]
    n_classes = len(class_dirs) + 1
    
    X = []
    y = []
    
    iterator = range(len(class_dirs))
    
    if verbose == 1:
        iterator = tqdm(iterator)
    
    for idx in iterator:
        
        c = class_dirs[idx]
        
        fnames = [os.path.join(c, f) for f in os.listdir(c)]
        
        if n_jobs > 1:
            X_c = _imap_unordered_bar(scipy.misc.imread, fnames, n_jobs)
        else:
            X_c = [scipy.misc.imread(f) for f in fnames]
        y_c = [idx] * len(X_c)
        
        X = X + X_c
        y = y + y_c
        
    if resize is not None:
        X = [scipy.misc.imresize(img, (resize, resize)) for img in X]
        
    X = np.array(X)
    y = np.array(y)
    
    X, y = shuffle(X, y, random_state = random_state)
    
    y = np.eye(n_classes - 1)[y]
    
    if split_test_train_ratio is None:
        return X, y
    else:
        X, Xt, y, yt = train_test_split(X, y, 
                                        test_size = split_test_train_ratio,
                                        random_state = random_state)
        return X, y, Xt, yt
    
    

    
    
    
def _imap_unordered_bar(func, args, n_processes = 10):
    try:
        p = Pool(n_processes)
        res_list = []
        with tqdm(total = len(args)) as pbar:
            for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
                pbar.update()
                res_list.append(res)
        pbar.close()
        p.close()
        p.join()
        return res_list
    except:
        p.close()
        p.join()
        print "error!!"