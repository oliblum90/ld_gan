from keras.datasets import mnist
from multiprocessing import Pool
import scipy.misc
import numpy as np
from utils import norm_img

def load_mnist(norm = True, resize = True):
    
    (X, y), (Xt, yt) = mnist.load_data()
    
    if resize:
        p = Pool(8)
        X = np.array(p.map(_resize, X))
        Xt = np.array(p.map(_resize, Xt))
    
    X = np.expand_dims(X, 3)
    Xt = np.expand_dims(Xt, 3)
    
    y = np.eye(np.max(y) + 1)[y]
    yt = np.eye(np.max(yt) + 1)[yt]
    
    if norm:
        X = norm_img.norm(X)
        Xt = norm_img.norm(Xt)
    
    return X, y, Xt, yt

def _resize(img):
    return scipy.misc.imresize(img, (32, 32))