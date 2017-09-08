import os
os.chdir("../")

import ld_gan

X, _ = ld_gan.data_proc.data_loader.load_data(2, n_jobs=10, resize=64)

_ = ld_gan.eval_gan.quick_score.inception_score("xf_11000_score.py", 650, X)