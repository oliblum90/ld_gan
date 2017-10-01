
import os
import numpy as np
from tqdm import tqdm
import scipy.misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

PATH_MNIST   = "/export/home/oblum/projects/ls_gan/data/mnist/jpg_32"
PATH_FLOWER  = "/export/home/oblum/projects/ld_gan/data/flowers_102/jpg_256"
PATH_CELEBA  = "/export/home/oblum/projects/ls_gan/data/celebA/jpg_64"
PATH_CELEBA_128 = "/export/home/oblum/projects/ls_gan/data/celebA/jpg_128/faces"
PATH_FLOWER_17 = "/export/home/oblum/projects/ls_gan/data/flowers_17/jpg_64"




def load_data_mp(path, 
                 split_test_train_ratio = None, 
                 random_state=42, 
                 n_jobs = 10,
                 resize = None):
    
    if path == -1:
        path = PATH_FLOWER_17
    if path == 0:
        path = PATH_MNIST
    elif path == 1:
        path = PATH_FLOWER
    elif path == 2:
        #path = PATH_CELEBA
        return load_celeba()
    
    print "load data from '{}'".format(path)
    
    class_dirs = [os.path.join(path, c) for c in os.listdir(path)]
    n_classes = len(class_dirs)
    
    # load single class
    if n_classes == 1:
        
        fnames = [os.path.join(class_dirs[0], f) for f in os.listdir(class_dirs[0])]
        y = len(os.listdir(class_dirs[0])) * [0]
        X = _imap_unordered_bar(scipy.misc.imread, fnames, n_jobs)
        
        X = np.array(X)
        y = np.array(y)

    else:
            
        X = _imap_bar(load_imgs, class_dirs, n_jobs)
        y = [len(os.listdir(class_dirs[c]))*[c] for c in range(len(class_dirs))]
        X = np.concatenate(X)
        y = np.concatenate(y)
    
    X, y = shuffle(X, y, random_state = random_state)
    
    y = np.eye(n_classes)[y]
    
    if split_test_train_ratio is None:
        return X, y
    else:
        X, Xt, y, yt = train_test_split(X, y, 
                                        test_size = split_test_train_ratio,
                                        random_state = random_state)
        return X, y, Xt, yt
    


def load_imgs(path):
    fnames = [os.path.join(path, n) for n in os.listdir(path)]
    return [scipy.misc.imread(n) for n in fnames]


def _load_single_celeba(lab_vec):
    #path = '/export/home/oblum/projects/ls_gan/data/celebA/jpg_64/'
    path = PATH_CELEBA
    path = os.path.join(path, lab_vec[0])
    img = scipy.misc.imread(path)
    lab_vec = np.array(lab_vec[1:]).astype('int')
    return [img, lab_vec]

def load_celeba(n_jobs=10):
    
    # load labels
    fname = '/export/home/oblum/projects/ls_gan/data/celebA/list_attr_celeba.txt'
    with open(fname, 'r') as f:
        label_str = f.read()
    label_str = label_str.replace('  ', ' ')
    lines = label_str.split('\r\n')
    labels = [s.split(' ') for s in lines[2:]][:-1]
    
    #p = Pool(n_jobs)
    #out = p.map(_load_single_celeba, labels)
    #out = _imap_unordered_bar(_load_single_celeba, labels, n_processes = n_jobs)
    out = [_load_single_celeba(l) for l in tqdm(labels)]
    
    x = np.array([o[0] for o in out])
    y = np.array([o[1] for o in out])
    
    y[y == -1] = 0
    
    return x, y






def load_data(path, 
              split_test_train_ratio = None, 
              random_state=42, 
              verbose = 0, 
              n_jobs = 1,
              resize = None):
    
    if path == -1:
        path = PATH_FLOWER_17
    if path == 0:
        path = PATH_MNIST
    elif path == 1:
        path = PATH_FLOWER
    elif path == 2:
        #path = PATH_CELEBA
        return load_celeba()
    
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
    
    

    
def _imap_bar(func, args, n_processes = 10):
    #try:
    p = Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list
    #except:
    #    p.close()
    #    p.join()
    #    print "error!!" 
    
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