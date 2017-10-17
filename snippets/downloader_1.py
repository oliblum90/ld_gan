import os
os.chdir("../")

import ld_gan
import numpy as np
import scipy.misc
import ld_gan.utils.utils as ld
from tqdm import tqdm
import urllib2


def save_img(line, path = "facescrub"):
    
    try:
    
        # get info
        split    = line.split("\t")
        name     = split[0]
        image_id = split[1]
        face_id  = split[2]
        url      = split[3]
        bbox     = split[4]
        x1, y1, x2, y2 = bbox.split(",")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # download
        f = urllib2.urlopen(url, timeout = 10)
        data = f.read()
        fname = "/tmp/tmp.jpg"
        with open(fname, "wb") as imgfile:
            imgfile.write(data)
        img = scipy.misc.imread(fname)

        # crop
        img_bb = img[y1:y2, x1:x2]

        # save imgs
        path_img = os.path.join(path, "imgs", name)
        path_cro = os.path.join(path, "crop", name)
        ld.mkdir(path_img)
        ld.mkdir(path_cro)
        fname_img = os.path.join(path_img, image_id + "_" + face_id + ".jpg")
        fname_cro = os.path.join(path_cro, image_id + "_" + face_id + ".jpg")
        scipy.misc.imsave(fname_img, img)
        scipy.misc.imsave(fname_cro, img_bb)
        
    except:
        
        with open("data/faceScrub/ERROR1.txt", 'a') as f:
            f.write(line+"\n")
    
    
    
if __name__ == "__main__":
    
    # init error_log
    with open("data/faceScrub/ERROR1.txt", 'w') as f:
        f.write("")
    
    with open("data/faceScrub/facescrub_actors.txt", 'r') as f:
        content = f.read()   
    lines = content.split("\n")


    #with open("data/faceScrub/facescrub_actresses.txt", 'r') as f:
    #    content = f.read()
    #lines = content.split("\n")


    for line in tqdm(lines):
        save_img(line)
    
    
    