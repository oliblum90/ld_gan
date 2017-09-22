import os
os.chdir("../")

import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import numpy as np

RAND_SEED = 42
cudnn.benchmark = True
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed_all(RAND_SEED)

X, Y = ld_gan.data_proc.data_loader.load_data(2)

project, epoch = "xc_11111_s10l.py", 10
enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
sampler = ld_gan.sample.nn_sampler_life(enc, X, Y, 1000)

out = sampler.next()

_ = ld_gan.eval_gan.quick_score.i_score(project, epoch, X, sampler)