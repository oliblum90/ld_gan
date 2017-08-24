from ld_gan.data_proc.transformer import np_to_tensor, tensor_to_np
import torch

def load_model(project, epoch, model_name):
        
    epoch_str = str(epoch).zfill(4)
    try:
        fname = "projects/" + project + "/model/" + model_name[0] + "_" + epoch_str + ".pth"
        model = torch.load(fname)
    except:
        fname = "projects/" + project + "/model/" + model_name + ".pth"
        model = torch.load(fname)
        
    print "loaded model '{}'".format(fname)
        
    return model


def apply_model(model, data):
        
    t_in  = np_to_tensor(data)
    t_out = model(t_in)
    arr_out = tensor_to_np(t_out)
    
    return arr_out