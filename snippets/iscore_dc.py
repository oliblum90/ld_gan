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


X, Y = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)

enc = ld_gan.utils.model_handler.load_model("xf_11111_s10l.py", 650, "enc")
gen = ld_gan.utils.model_handler.load_model("xf_11111_s10l.py", 650, "gen")
dis = ld_gan.utils.model_handler.load_model("xf_11111_s10l.py", 650, "dis")

enc = torch.nn.DataParallel(enc, device_ids=[0,1]).cuda()
gen = torch.nn.DataParallel(gen, device_ids=[0,1]).cuda()

from ld_gan.utils.model_handler import apply_model
from ld_gan.utils.model_handler import apply_models
from tqdm import tqdm
from time import time

N_Z_ANCHORS = 100
N_ENC_SPACE_DIMS = 265
N_Z_POTENTIALS_PER_A = 200
N_INTERPOL = 8

z_enc = apply_model(enc, X)
idxs = range(len(z_enc))
z_anchors = z_enc[np.random.choice(idxs, size=N_Z_ANCHORS, replace=False)]

n_potentials = N_Z_POTENTIALS_PER_A * N_Z_ANCHORS
z_enc = np.tile(z_enc, (n_potentials / len(z_enc) + 1, 1))[:n_potentials]
z_potentials = np.split(z_enc, len(z_anchors))

t = time()

z_potentials = np.split(z_enc, len(z_anchors))

zs = []
for i in range(N_INTERPOL):
    for z_anchor, z_potential in zip(z_anchors, z_potentials):
        z_anchor = np.tile(z_anchor, (N_Z_POTENTIALS_PER_A, 1))
        f1 = i/float(N_INTERPOL-1)
        f2 = 1 - i/float(N_INTERPOL-1)
        zs.append(f1*z_anchor + f2*z_potential)
zs = np.concatenate(zs)


time()-t

t = time()

d = apply_models(zs, 3000, gen, dis)

print time()-t
