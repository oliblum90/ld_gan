import os
from tqdm import tqdm
import numpy as np
import torch
import ld_gan
import shutil
import scipy.misc

import visualize
from data_proc.transformer import np_to_tensor, tensor_to_np
from utils.init_project import init_project, save_setup
from ld_gan.utils.log_time import log_time
from ld_gan.utils.logging import remove_nans, log_host_name
import __main__ as main

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
                 gen_tsne_step     = 10,
                 save_model_step   = 50,
                 gen_iscore_step   = 100,
                 bs_tsne_pts       = None,
                 callbacks         = [],
                 gpu_id            = None):
        
        # set class variables
        if project_name is None:
            project_name =  os.path.join("projects", main.__file__)
        self.project_name     = project_name
        self._path_log        = os.path.join(project_name, "log")
        self._path_model      = os.path.join(project_name, "model")
        self._path_gen_img    = os.path.join(project_name, "generated_img")
        self._path_incpt_imgs = os.path.join(project_name, "incept_imgs")
        self._path_hist_tsne  = os.path.join(project_name, "hist_tsne")
        
        self.gen = gen
        self.dis = dis
        self.enc = enc
        
        self.sampler          = sampler
        self.train_ops        = train_ops
        self.callbacks        = callbacks
        
        self._gen_img_step    = gen_img_step
        self._gen_tsne_step   = gen_tsne_step
        self._save_model_step = save_model_step
        self._gen_iscore_step = gen_iscore_step
        self.n_samples        = n_samples
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.iters_per_epoch  = n_samples / batch_size
        
        self.epoch_losses     = []
        self.bs_tsne_pts      = bs_tsne_pts
        
        # init project
        init_project(project_name, ask_before_del = ask_before_del)
        save_setup(project_name)
        init_project(self._path_log)
        init_project(self._path_model)
        init_project(self._path_gen_img)
        init_project(self._path_hist_tsne)
        init_project(self._path_incpt_imgs)
        
        # init log
        self._init_log()
        np.savetxt(os.path.join(self._path_log, "iters_per_epoch"), 
                   np.array([self.iters_per_epoch]))
        np.savetxt(os.path.join(self._path_log, "batch_size"), 
                   np.array([self.batch_size]))
        np.savetxt(os.path.join(self._path_log, "n_samples"), 
                   np.array([self.n_samples]))
        log_host_name(self._path_log)
        if gpu_id is not None:
            np.savetxt(os.path.join(self._path_log, "gpu_id.txt"),
                       np.array([gpu_id]))
        
        # save models
        torch.save(self.gen, os.path.join(self._path_model, "gen.pth"))
        torch.save(self.dis, os.path.join(self._path_model, "dis.pth"))
        torch.save(self.enc, os.path.join(self._path_model, "enc.pth"))
        
        # init iscore model
        self.i_score = ld_gan.eval_gan.InceptionScore()
        self.i_score_list = []
        
        
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
        
        fname = os.path.join(path, "e_" + epoch_str + ".pth")
        torch.save(self.enc, fname)
            

    def generate_imgs(self, fname = None):
        
        print "generate test imgs..."
        
        X, Y, Z, _, _, _, _, _ = self.sampler.next()
        
        Z_exact = ld_gan.utils.model_handler.apply_model(self.enc, X)
        x = ld_gan.utils.model_handler.apply_model(self.gen, Z)
        x_exact = ld_gan.utils.model_handler.apply_model(self.gen, Z_exact)
        
        if fname is not None:
            
            fname_fake = os.path.join(self._path_gen_img, fname + "_fake.png")
            visualize.save_g_imgs(fname_fake, x_exact)
            
            fname_fake = os.path.join(self._path_gen_img, fname + "_mean_fake.png")
            visualize.save_g_imgs(fname_fake, x)
            
            fname_real = os.path.join(self._path_gen_img, fname + "_real.png")
            visualize.save_g_imgs(fname_real, X)
            
        else:
            
            return x, X
        
        
    def save_tsne_hist(self, e_str, n_f_vecs = 4000, n_tsne_imgs = 30):
        
        print "generate z-histogram and tsne visualization..."
        
        n_iters = n_f_vecs / self.batch_size + 1
        f_vecs, imgs = [], []
        ys = []
        for step in range(n_iters):
            X, Y, Z, _, _, _, _, _ = self.sampler.next()
            f_vecs.append(Z)
            imgs.append(X)
            ys.append(Y)
        X = np.concatenate(imgs)[:n_f_vecs]
        Z = np.concatenate(f_vecs)[:n_f_vecs]
        Y = np.concatenate(ys)[:n_f_vecs]
        Z = ld_gan.utils.model_handler.apply_model(self.enc, X, self.batch_size)
            
        fname = os.path.join(self._path_hist_tsne, e_str + "_hist_tsne.png")
        visualize.plot_hist_and_tsne(Z, 
                                     y = Y,
                                     imgs = X, 
                                     fname = fname,
                                     n_clusters = n_tsne_imgs,
                                     n_pts_tsne = n_f_vecs,
                                     project = self.project_name,
                                     epoch_str = e_str)
        
                
    def get_inception_score(self, e_str, n_samples=50000):
        
        print "generate incept score samples..."
        n_iters = int((n_samples / self.batch_size) + 1)
        imgs = []
        ys = []
        for it in tqdm(range(n_iters)):
            _, Y, Z, _, _, _, _, _ = self.sampler.next()
            imgs.append(ld_gan.utils.model_handler.apply_model(self.gen, Z))
            ys.append(Y)
        imgs = np.concatenate(imgs)
        ys = np.concatenate(ys)
        
        print "compute inception score..."
        score = self.i_score.score(imgs, batch_size=32)
        self.i_score_list.append(score)
        
        # init log
        fname_log = os.path.join(self._path_log, "iscore.txt")
        if not os.path.isfile(fname_log):
            print "creating iscore log-file..."
            header = "iscore"
            with open(fname_log, 'w') as f:
                f.write(header)
        
        # write log
        with open(fname_log, 'a') as f:
            f.write("\n" + str(score))

        print "score: {}".format(score)
            
        print "save new incept imgs..."
        path = os.path.join(self._path_incpt_imgs, e_str)
        os.mkdir(path)
        for idx in range(len(imgs)):
            fname = str(idx).zfill(6) + "_" + str(ys[idx]).zfill(3) + ".jpg"
            fname = os.path.join(path, fname)
            scipy.misc.imsave(fname, imgs[idx])
                
            
        
    def _show_training_status(self, epoch):
        
        self.epoch_losses = remove_nans(self.epoch_losses)
        losses = np.mean(np.array(self.epoch_losses), axis = 0)
        self.epoch_losses = []
        names = [to.__class__.__name__ for to in self.train_ops]
        names = [n.ljust(10) for n in names]
        
        print "EPOCH: {}, ({})".format(epoch, main.__file__)
        print "--------------------------------------------------------"
        for l, n in zip(losses, names):
            print n, ": ", l
        print "--------------------------------------------------------"
        
        
    def train(self):
        
        print "\nstart training..."
                
        for epoch in range(self.n_epochs + 1):
            
            e_str = str(epoch).zfill(4)
                        
            for it in tqdm(range(self.iters_per_epoch)):
                
                X, Y, Z, i1, i2, i3, z_all, _ = self.sampler.next()
                X, Y, Z = np_to_tensor(X, Y, Z)
                
                log_time("train")
                losses = [op.train(X,Y,Z,i1,i2,i3,z_all) if it % op.freq == 0 else -1000
                          for op in self.train_ops]
                log_time("train")
            
                self._write_log(losses)
            
            self._show_training_status(epoch)
            
            # run callbacks
            for cb in self.callbacks:
                if epoch % cb.run_every_nth_epoch == 0:
                    cb.run()
            
            # save inception score
            if epoch % self._gen_iscore_step == 0:
                self.get_inception_score(e_str)
            
            # save generated imgs
            if epoch % self._gen_img_step == 0: 
                self.generate_imgs(fname = e_str)
            
            # save tsne and hist
            if epoch % self._gen_tsne_step == 0:
                self.save_tsne_hist(e_str)

            # save model
            if epoch % self._save_model_step == 0:
                self.save_model(e_str)
    
    