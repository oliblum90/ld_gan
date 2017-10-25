import os
os.chdir("../")

import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import numpy as np
import sys
import time


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    t = int(sys.argv[2])
    
    with torch.cuda.device(gpu_id):
        
        a = torch.rand(100,100,100,100)
        time.sleep(t)