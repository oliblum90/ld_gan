import urllib
import tarfile
import os
import numpy as np

PATH_DATA_RAW   = "data"
FNAME_EXTRACTED = "cifar-10-batches-py"
URL_CIFAR_10    = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def _download(url):
    fname = os.path.join(PATH_DATA_RAW, url.split('/')[-1])
    urllib.urlretrieve(url, fname)
    
    
def _extract(fname):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(PATH_DATA_RAW)
    tar.close()


def _unpickle(fname):
    import cPickle
    with open(fname, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def _load_data_from_batches(train_batches):
    data = []
    labels = []
    for data_batch_i in train_batches:
        path = os.path.join(PATH_DATA_RAW, FNAME_EXTRACTED)
        d = _unpickle(os.path.join(path, data_batch_i))
        data.append(d['data'])
        labels.append(np.array(d['labels']))
    # Merge training batches on their first dimension
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    length = len(labels)

    data = data.reshape(length, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)
    
    return data, labels


def _load_data():
    X, y = _load_data_from_batches(
        ["data_batch_{}".format(i) for i in range(1, 6)]
    )

    Xt, yt = _load_data_from_batches(["test_batch"])
    
    return X, y, Xt, yt
    
    
def load_cifar10(norm = True):
    
    print "download..."
    url = URL_CIFAR_10
    fname = os.path.join(PATH_DATA_RAW, url.split('/')[-1])
    if os.path.isfile(fname):
        pass
    else:
        _download(url)
        
    print "extract..."
    path = os.path.join(PATH_DATA_RAW, FNAME_EXTRACTED)
    if os.path.isdir(path):
        pass
    else:
        _extract(fname)
        
    print "load data..."
    X, y, Xt, yt = _load_data()
    y = np.eye(np.max(y) + 1)[y]
    yt = np.eye(np.max(yt) + 1)[yt]
    
    # if norm:
        # X = (X.astype('float32') - 127.5) / 255.
        # Xt = (Xt.astype('float32') - 127.5) / 255.
    
    return X, y, Xt, yt
    
    
    