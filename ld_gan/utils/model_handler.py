import ld_gan
from ld_gan.data_proc.transformer import np_to_tensor, tensor_to_np
import torch
import numpy as np

def load_model(project, epoch, model_name, test_mode = True):
        
    epoch_str = str(epoch).zfill(4)
    try:
        fname = "projects/" + project + "/model/" + model_name[0] + "_" + epoch_str + ".pth"
        model = torch.load(fname)
    except:
        fname = "projects/" + project + "/model/" + model_name + ".pth"
        model = torch.load(fname)
        
    if test_mode:
        model.eval()
        
    print "loaded model '{}'".format(fname)
        
    return model


def load_parallel_model(project, epoch, model_name):
    
    temp = load_model(project, epoch, model_name)
    
    if model_name == "enc":
        model = ld_gan.models.enc.Enc(n_features = 256)
    elif model_name == "gen":
        model = ld_gan.models.gen.Gen(latent_size = 256)
    elif model_name == "dis":
        model = ld_gan.models.dis.Dis()
    
    sd = temp.state_dict()
    
    for key in sd.keys():
        sd[key[7:]] = sd.pop(key)
        
    model.load_state_dict(sd)
    model.cuda()
    
    return model
    



def apply_model(model, data, batch_size = None, cpu = False):
    
    if batch_size is None:
        t_in  = np_to_tensor(data)
        if cpu:
            t_in = t_in.cpu()
            model.cpu()
        t_out = model(t_in)
        arr_out = tensor_to_np(t_out)
        
    else:
        n_batches = len(data) / batch_size
        data = np.array_split(data, n_batches)
        arr_out = []
        for d in data:
            t_in  = np_to_tensor(d)
            if cpu:
                t_in = t_in.cpu()
                model.cpu()
            t_out = model(t_in)
            arr_out.append(tensor_to_np(t_out))
        arr_out = np.concatenate(arr_out)
    
    return arr_out


def apply_models(data, batch_size, *models):
    
    if batch_size == None:
        data = [data]
    elif batch_size >= len(data):
        data = [data]
    else:
        n_batches = len(data) / batch_size
        data = np.array_split(data, n_batches)
    
    arr_out = []
    for d in data:
        t  = np_to_tensor(d)
        for m in models:
            t = m(t)
        arr_out.append(tensor_to_np(t))
    arr_out = np.concatenate(arr_out)
    
    return arr_out

