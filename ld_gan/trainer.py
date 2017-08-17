import os
from tqdm import tqdm
import numpy as np

import visualize
from data_proc.transformer import np_to_tensor, tensor_to_np
from utils.init_project import init_project, save_setup


class Trainer:
    
    def __init__(self,
                 gen, 
                 dis,
                 enc,
                 train_ops,
                 sampler,
                 n_samples,
                 project_name      = None,
                 n_epochs          = 100,
                 batch_size        = 128,
                 ask_before_del    = False):
        
        # set class variables
        self.gen = gen
        self.dis = dis
        self.enc = enc
        
        self.sampler = sampler
        self.train_ops = train_ops
        
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.iters_per_epoch = n_samples / batch_size
        
        self.epoch_losses = []
        
        if project_name is None:
            import __main__ as main
            project_name =  os.path.join("projects", main.__file__)
        self.project_name    = project_name
        self._path_log       = os.path.join(project_name, "log")
        self._path_model     = os.path.join(project_name, "model")
        self._path_gen_img    = os.path.join(project_name, "generated_img")
        self._path_hist_tsne  = os.path.join(project_name, "hist_tsne")
        
        # init project
        init_project(project_name, ask_before_del = ask_before_del)
        save_setup(project_name)
        init_project(self._path_log)
        init_project(self._path_model)
        init_project(self._path_gen_img)
        init_project(self._path_hist_tsne)
        
        # init log
        self._init_log()
        
        
        
        
    def _init_log(self):
        
        fname = os.path.join(self._path_log, "logs.txt")
        header = " ".join([to.__class__.__name__ for to in self.train_ops])
        with open(fname, 'w') as f:
            f.write(header)
        
        
    def _write_log(self, losses):
        
        self.epoch_losses.append(losses)
        fname = os.path.join(self._path_log, "logs.txt")
        line = " ".join([str(l) for l in losses])
        with open(fname, 'a') as f:
            f.write("\n" + line)


    def generate_imgs(self, fname = None):
        
        Z, _, X, _ = self.sampler.next()
        Z = np_to_tensor(Z)
        x = self.gen(Z)
        x = tensor_to_np(x)
        
        if fname is not None:
            
            fname_fake = os.path.join(self._path_gen_img, fname + "_fake.png")
            visualize.save_g_imgs(fname_fake, x)
            
            fname_real = os.path.join(self._path_gen_img, fname + "_real.png")
            visualize.save_g_imgs(fname_real, X)
            
        else:
            
            return x, X
        
        
    def _show_training_status(self, epoch):
        
        losses = np.mean(np.array(self.epoch_losses), axis = 0)
        self.epoch_losses = []
        names = [to.__class__.__name__ for to in self.train_ops]
        
        print "EPOCH: {}".format(epoch)
        print "--------------------------------------------------------"
        for l, n in zip(losses, names):
            print n, "\t: ", l
        print "--------------------------------------------------------"
        
        
    def train(self):
                
        for epoch in range(self.n_epochs):
            
            e_str = str(epoch).zfill(4)
                        
            for iteration in tqdm(range(self.iters_per_epoch)):
                
                Z, _, X, _ = self.sampler.next()
                Z, X = np_to_tensor(Z, X)
                
                losses = [op.train(X, Z) for op in self.train_ops]
            
                self._write_log(losses)
            
            self._show_training_status(epoch)
            self.generate_imgs(fname = e_str)
        
    
    