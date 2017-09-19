import os
import numpy as np
import socket


def remove_nans(logs):
    
    logs = np.array(logs)
    
    for i in range(10):
        for i1 in range(logs.shape[0]):
            for i2 in range(logs.shape[1]):
                logs[i1, i2] = logs[i1, i2] if logs[i1, i2]!=-1000 else logs[i1, i2-1]
                
    return logs


def log_host_name(path):

    fname = os.path.join(path, "host_name.txt")
    with open(fname, 'w') as f:
        f.write(socket.gethostname())