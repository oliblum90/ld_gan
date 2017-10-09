import os
os.chdir("../")

import ld_gan
import numpy as np
import sys
from tqdm import tqdm
import scipy.misc


if __name__ == "__main__":

    PATH = "eval_imgs/xf_111v1.py/"
    BATCH_SIZE = 32


    imgs_fake = [scipy.misc.imread(os.path.join(PATH, n)) \
                 for n in tqdm(os.listdir(PATH)) if ".png" in n]
    imgs_fake = imgs_fake
    score = ld_gan.eval_gan.get_inception_score(imgs_fake, splits=2)

    print "finished"
    print "----------------------------------------"
    print "score = ", score

    np.savetxt(os.path.join(PATH, "inception_score_tf.txt"), np.array([score]))