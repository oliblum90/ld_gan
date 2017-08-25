import os
from tqdm import tqdm
import numpy as np
import torch

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
                 ask_before_del    = False,
                 gen_img_step      = 5,
                 gen_tsne_step     = 25,
                 save_model_step   = 50,
                 trainable_enc     = False):
        
        # set class variables
        if project_name is None:
            import __main__ as main
            project_name =  os.path.join("projects", main.__file__)
        self.project_name     = project_name
        self._path_log        = os.path.join(project_name, "log")
        self._path_model      = os.path.join(project_name, "model")
        self._path_gen_img    = os.path.join(project_name, "generated_img")
        self._path_hist_tsne  = os.path.join(project_name, "hist_tsne")
        
        self.gen = gen
        self.dis = dis
        self.enc = enc
        
        self.sampler          = sampler
        self.train_ops        = train_ops
        self.trainable_enc    = trainable_enc
        
        self._gen_img_step    = gen_img_step
        self._gen_tsne_step   = gen_tsne_step
        self._save_model_step = save_model_step
        self.n_samples        = n_samples
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.iters_per_epoch  = n_samples / batch_size
        
        self.epoch_losses     = []
        
        # init project
        init_project(project_name, ask_before_del = ask_before_del)
        save_setup(project_name)
        init_project(self._path_log)
        init_project(self._path_model)
        init_project(self._path_gen_img)
        init_project(self._path_hist_tsne)
        
        # init log
        self._init_log()
        np.savetxt(os.path.join(self._path_log, "iters_per_epoch"), 
                   np.array([self.iters_per_epoch]))
        np.savetxt(os.path.join(self._path_log, "batch_size"), 
                   np.array([self.batch_size]))
        np.savetxt(os.path.join(self._path_log, "n_samples"), 
                   np.array([self.n_samples]))
        
        # save hist / t-sne
        if trainable_enc == False:
            self.save_tsne_hist(fname = "0")
        
        # save models
        torch.save(self.gen, os.path.join(self._path_model, "gen.pth"))
        torch.save(self.dis, os.path.join(self._path_model, "dis.pth"))
        torch.save(self.enc, os.path.join(self._path_model, "enc.pth"))
        
        
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

            
    def save_model(self, epoch_str):
        
        print "save model..."
        
        path = os.path.join(self.project_name, "model")
        
        fname = os.path.join(path, "g_" + epoch_str + ".pth")
        torch.save(self.gen, fname)
        
        fname = os.path.join(path, "d_" + epoch_str + ".pth")
        torch.save(self.dis, fname)
        
        if self.trainable_enc:
            fname = os.path.join(path, "e_" + epoch_str + ".pth")
            torch.save(self.enc, fname)
            

    def generate_imgs(self, fname = None):
        
        print "generate test imgs..."
        
        Z, _, X, _ = self.sampler.next()
        
        if self.trainable_enc:
            Z = self.enc(np_to_tensor(X))
        else:
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
        
        
    def save_tsne_hist(self, fname, n_f_vecs = 1000, n_tsne_imgs = 30):
        
        print "generate z-histogram and tsne visualization..."
        
        n_iters = n_f_vecs / self.batch_size + 1
        f_vecs, imgs = [], []
        for step in range(n_iters):
            Z, _, X, _ = self.sampler.next()
            f_vecs.append(Z)
            imgs.append(X)
        X = np.concatenate(imgs)[:n_f_vecs]
        Z = np.concatenate(f_vecs)[:n_f_vecs]

        if self.trainable_enc:
            Z = tensor_to_np(self.enc(np_to_tensor(X)))
            
        fname = os.path.join(self._path_hist_tsne, fname + "_hist_tsne.png")
        visualize.plot_hist_and_tsne(Z, 
                                     imgs = X, 
                                     fname = fname,
                                     n_clusters = n_tsne_imgs,
                                     n_pts_tsne = n_f_vecs)
        
        
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
        
        print "\nstart training..."
                
        for epoch in range(self.n_epochs + 1):
            
            e_str = str(epoch).zfill(4)
                        
            for iteration in tqdm(range(self.iters_per_epoch)):
                
                Z, _, X, _ = self.sampler.next()
                Z, X = np_to_tensor(Z, X)
                
                losses = [op.train(X, Z) for op in self.train_ops]
            
                self._write_log(losses)
            
            self._show_training_status(epoch)
            
            # save generated imgs
            if epoch % self._gen_img_step == 0: 
                self.generate_imgs(fname = e_str)
            
            # save tsne and hist
            if epoch % self._gen_tsne_step == 0:
                self.save_tsne_hist(fname = e_str)

            # save model
            if epoch % self._save_model_step == 0:
                self.save_model(e_str)
    
    