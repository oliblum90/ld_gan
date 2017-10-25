
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
PATH_BIRDS = "/export/home/oblum/projects/ld_gan/data/birds/images"
PATH_FLOWER_TR  = "/export/home/oblum/projects/ld_gan/data/flowers_102/jpg_train_256"
PATH_FLOWER_TE  = "/export/home/oblum/projects/ld_gan/data/flowers_102/jpg_test_256"
PATH_PETS = "/export/home/oblum/projects/ld_gan/data/pets/cropped"
PATH_FS_1 = "/export/home/oblum/projects/ld_gan/data/faceScrub/imgs/"
PATH_FS_2 = "/export/home/oblum/projects/ld_gan/data/faceScrub/crop/"

def load_data(path,
              random_state=42, 
              verbose = 1, 
              n_jobs = 1,
              resize = None,
              all_img_in_one_dir=False,
              test_train = None, 
              gray_to_rgb = False, 
              file_type=".jpg"):
    
    if path == -1:
        path = PATH_FLOWER_17
    if path == 0:
        path = PATH_MNIST
    elif path == 1:
        path = PATH_FLOWER
    elif path == 10:
        path = PATH_FLOWER_TR
    elif path == 11:
        path = PATH_FLOWER_TE
    elif path == 2:
        path = PATH_CELEBA
    elif path == 3:
        path = PATH_BIRDS
    elif path == 4:
        path = PATH_PETS
    elif path == 51:
        path = PATH_FS_1
    elif path == 52:
        path = PATH_FS_2        
        
    print "load data from '{}'".format(path)
    
    if all_img_in_one_dir:
        class_dirs = [path]
    else:
        class_dirs = [os.path.join(path, c) for c in os.listdir(path) \
                      if os.path.isdir(os.path.join(path, c))]
    
    n_classes = len(class_dirs) + 1
    
    X = []
    y = []
    
    iterator = range(len(class_dirs))
    
    if verbose >= 1:
        iterator = tqdm(iterator)
    
    for idx in iterator:
        
        c = sorted(class_dirs)[idx]

        if test_train is None:
            fnames =[os.path.join(c, f) for f in sorted(os.listdir(c))]
        elif test_train == "test":
            fnames =[os.path.join(c, f) for f in sorted(os.listdir(c)) if "_test" in f]
        elif test_train == "train":
            fnames =[os.path.join(c, f) for f in sorted(os.listdir(c)) if "_test" not in f]
        else:
            print "UNDEFINED!!!"
            
        if file_type is not None:
            fnames = [fname for fname in fnames if file_type in fname]
        
        iterator_2 = sorted(fnames)
        if verbose == 2:
            iterator_2 = tqdm(sorted(fnames))
                
        if n_jobs > 1:
            X_c = _imap_unordered_bar(scipy.misc.imread, fnames, n_jobs)
        else:
            X_c = [scipy.misc.imread(f) for f in iterator_2]
        y_c = [idx] * len(X_c)
        
        X = X + X_c
        y = y + y_c
        
    if resize is not None:
        X = [scipy.misc.imresize(img, (resize, resize)) for img in X]
        
    if gray_to_rgb:
        for idx_img in range(len(X)):
            if X[idx_img].ndim==2:
                X[idx_img] = np.array([X[idx_img],X[idx_img],X[idx_img]]).transpose(1,2,0)
        
    X = np.array(X)
    y = np.array(y)
    
    # X, y = shuffle(X, y, random_state = random_state)
    
    y = np.eye(n_classes - 1)[y]
    
    return X, y
    
    

    
import os
import torch.utils.data as data
import torch
import numpy as np
import scipy.misc

def live_loader(path, batch_size, file_type = '.jpg', resize=128):

    def listdir(path, file_type = None):
        if file_type is None:
            file_type = ""

        fnames = []
        dirs = [path]
        while len(dirs) > 0:
            path = dirs.pop()
            content = [os.path.join(path, n) for n in os.listdir(path)]
            dirs = [c for c in content if os.path.isdir(c)] + dirs
            fnames += [c for c in content if file_type in c]

        return sorted(fnames)

    
    class CustomDataset(data.Dataset):
        def __init__(self, path, file_type='.jpg'):
            
            # get fnames
            self.fnames = listdir(path, file_type)
            
            # get label vec
            classes = {}
            for c, c_name in enumerate(sorted(os.listdir(path))):
                classes[c_name] = c
            self.labels = np.array([classes[name.split("/")[-2]] for name in self.fnames])
            
        def __getitem__(self, index):
            
            x = scipy.misc.imread(self.fnames[index])
            x = scipy.misc.imresize(x, (resize, resize))
            x = x.transpose(2, 0, 1)
            y = self.labels[index]
            return x, y
            
        def __len__(self):
            # You should change 0 to the total size of your dataset.
            return len(self.fnames) 

    # Then, you can just use prebuilt torch's data loader. 
    train_loader = torch.utils.data.DataLoader(dataset=CustomDataset(path, file_type),
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=1)
    
    return train_loader
    
    
