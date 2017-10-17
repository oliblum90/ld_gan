import os
os.chdir("../")

import ld_gan.utils.utils as ld
import scipy.misc
from tqdm import tqdm



if __name__ == "__main__":
    
    fnames = ld.listdir("data/faceScrub/imgs/", ".jpg", recrusive=True)
    
    for fname in tqdm(fnames):
        
        try:
            img = scipy.misc.imread(fname)
        
        except:
            os.remove(fname)