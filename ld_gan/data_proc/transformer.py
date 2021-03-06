import numpy as np
import torch
from norm_img import norm, un_norm
from torch.autograd import Variable


def np_to_tensor(*args, **kwargs):

    try:
        normalize = kwargs["normalize"]
    except:
        normalize = True
    
    tensors = []
    
    for arr in args:
        
        if arr.ndim == 4 and arr.shape[-1] != 1:
            arr = arr.transpose(0, 3, 1, 2)
            if normalize:
                arr = norm(arr)
            
        if arr.ndim == 2:
            arr = arr.reshape((arr.shape[0], arr.shape[1], 1, 1))
            
        if arr.ndim == 1:
            t = Variable(torch.from_numpy(arr).cuda().long())
        else:
            t = Variable(torch.from_numpy(arr).cuda()).float()
        
        tensors.append(t)
        
    return tensors if len(tensors)>1 else tensors[0]
    
    

def tensor_to_np(t, normalize=True):
    
    try:
        arr = (t.data).cpu().numpy()
    except:
        arr = (Variable(t).data).cpu().numpy()
    
    if arr.ndim == 4 and arr.shape[-1] != 1:
        if normalize:
            arr = un_norm(arr)
        arr = arr.transpose(0, 2, 3, 1)
        
    arr = np.squeeze(arr)
        
    return arr



def transform(*args, **kwargs):
    
    try:
        normalize = kwargs["normalize"]
    except:
        normalize = True
            
    if type(args[0]) is np.ndarray:
        return np_to_tensor(*args, normalize=normalize)
        
    else:
        np_arrs = []
        for arg in args:
            np_arrs.append(tensor_to_np(arg, normalize=normalize))
        return np_arrs if len(np_arrs)>1 else np_arrs[0] 
        
        
        
        
        