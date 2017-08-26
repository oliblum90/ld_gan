import numpy as np
import torch
from norm_img import norm, un_norm
from torch.autograd import Variable


def np_to_tensor(*args):

    tensors = []
    
    for arr in args:
        
        if arr.ndim == 4 and arr.shape[-1] != 1:
            arr = arr.transpose(0, 3, 1, 2)
            arr = norm(arr)
            
        if arr.ndim == 2:
            arr = arr.reshape((arr.shape[0], arr.shape[1], 1, 1))
            
        if arr.ndim == 1:
            t = Variable(torch.from_numpy(arr).cuda().long())
        else:
            t = Variable(torch.from_numpy(arr).cuda()).float()
        
        tensors.append(t)
        
    return tensors if len(tensors)>1 else tensors[0]
    
    

def tensor_to_np(t):
        
    try:
        arr = (t.data).cpu().numpy()
    except:
        arr = (Variable(t).data).cpu().numpy()
    
    if arr.ndim == 4 and arr.shape[-1] != 1:
        arr = un_norm(arr)
        arr = arr.transpose(0, 2, 3, 1)
        
    arr = np.squeeze(arr)
        
    return arr
        
            
        