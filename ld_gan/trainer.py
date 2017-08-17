import os
from tqdm import tqdm
import numpy as np

import visualize
from data_proc.transformer import np_to_tensor, tensor_to_np

class Trainer:
    
    def __init__(self,
                 gen, 
                 dis,
                 enc,
                 train_ops,
                 sampler,
                 n_samples,
                 n_epochs   = 100,
                 batch_size = 128,
                 ask_before_del = False):
        
        # init project
        #if project_name is None:
        #    import __main__ as main
        #    project_name =  main.__file__
        #project_name = os.path.join("projects", project_name)
        #init_project(project_name, ask_before_del = ask_before_del)
        #save_setup(project_name)
        #init_project(os.path.join(project_name, "generated_img"))
        #init_project(os.path.join(project_name, "model"))
        #init_project(os.path.join(project_name, "log"))
        #init_project(os.path.join(project_name, "hist_tsne"))
        
        # init log
        #self._init_log()
        
        # set class variables
        self.gen = gen
        self.dis = dis
        self.enc = enc
        
        self.sampler = sampler
        self.train_ops = train_ops
        
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.iters_per_epoch = n_samples / batch_size
        
        
    def train(self):
                
        for epoch in range(self.n_epochs):
                        
            for iteration in tqdm(range(self.iters_per_epoch)):
                
                Z, _, X, _ = self.sampler.next()
                Z, X = np_to_tensor(Z, X)
                
                losses = [op.train(X, Z) for op in self.train_ops]
            
                #self.write_log(losses)
            
            print epoch
            imgs = tensor_to_np(self.gen(Z))
            print imgs.shape
            visualize.disp(imgs[:10])
        
        
    def _init_log(self):
        
        fname = os.path.join(self._path_log, "logs.txt")
        header = " ".join([to.__class__.__name__ for to in self.train_ops])
        with open(fname, 'w') as f:
            f.write(header)
        
        
    def _write_log(self, losses):
        
        fname = os.path.join(self._path_log, "logs.txt")
        line = " ".join([str(l) for l in losses])
        with open(fname, 'a') as f:
            f.write("\n" + line)
        
        
        
        
        
        
    
    