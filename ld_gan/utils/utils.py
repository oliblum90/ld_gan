import os
import numpy as np
import scipy.misc


def listdir(path, file_type = None, recrusive=False):
    if file_type is None:
        file_type = ""
    if recrusive:
        fnames = []
        dirs = [path]
        while len(dirs) > 0:
            path = dirs.pop()
            content = [os.path.join(path, n) for n in os.listdir(path)]
            dirs = [c for c in content if os.path.isdir(c)] + dirs
            fnames += [c for c in content if file_type in c]
    else:
        fnames = [os.path.join(path, n) for n in os.listdir(path) if file_type in n]
        
    return sorted(fnames)


def disp(img_list, title_list = None, fname = None, figsize=None):
    """
    display a list of images
    """
    import matplotlib.pylab as plt

    if len(img_list) > 20:
        img_list = [img_list]
    
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize, dpi=180)

    for idx, img in enumerate(img_list):

        plt.subplot(1, len(img_list), idx+1)
        if title_list is not None:
            plt.title(title_list[idx])
        
        if type(img) == str:
            img = scipy.misc.imread(img)
        img_show = img.copy()
        
        if len(img_show.shape) == 2:
            plt.imshow(img_show.astype(np.uint8), 
                       vmax = img.max(), vmin = img.min(),
                       cmap='gray')
        else:
            plt.imshow(img_show.astype(np.uint8), 
                       vmax = img.max(), vmin = img.min())
            
        plt.axis("off")
    
    if fname is not None:
        plt.savefig(fname)
        
    plt.show()
    
    
def extract(fname):
    
    if fname.split('.')[-1] == "tar":
        os.system("tar -xvf " + fname)
        
    elif fname.split('.')[-1] == "gz":
        os.system("tar -xzvf " + fname)
    elif fname.split('.')[-1] == "zip":
        import zipfile
        zip_ref = zipfile.ZipFile(fname, 'r')
        zip_ref.extractall(fname[:-4])
        zip_ref.close()
    else:
        print "unknown file type"
        sys.exit()
        
    path_from = fname.split('/')[-1].split(".")[0]
    path_to = os.path.dirname(os.path.abspath(fname))
    os.system("mv " + path_from + " " + path_to)

        
def mkdir(dir_name):
    dir_split = dir_name.split("/")
    d_current = ""
    for d in dir_split:
        d_current = os.path.join(d_current, d)
        if not os.path.isdir(d_current):
            os.mkdir(d_current)

def load_imgs(fnames, resize=None, gray_to_rgb=False):
    imgs = [scipy.misc.imread(fname) for fname in fnames]
    if resize is not None:
        imgs = [scipy.misc.imresize(img, (resize, resize)) for img in imgs]
    if gray_to_rgb:
        for img_idx in range(len(imgs)):
            if imgs[img_idx].ndim == 2:
                imgs[img_idx] = np.array([imgs[img_idx], 
                                          imgs[img_idx], 
                                          imgs[img_idx]]).transpose(1,2,0)
    return np.array(imgs)

