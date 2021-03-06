import os
import torchvision
from torchvision.models.inception import model_urls
import numpy as np
import scipy.misc
from torchvision import models, transforms
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import ld_gan
import __main__ as main


class InceptionScore:
    
    def __init__(self, 
                 sampler=None, 
                 n_samples=None, 
                 enc=None, 
                 gen=None,
                 model=None,
                 run_every_nth_epoch=5,
                 real_data_score=None):
        """
        Class for the computation of the inception score
        """
        
        # set class variables
        self.sampler = sampler
        self.n_samples = n_samples
        self.enc = enc
        self.gen = gen
        self.run_every_nth_epoch = run_every_nth_epoch
        self.real_data_score = real_data_score
        
        # load model
        if model is None:
            try:
                self.incept = torchvision.models.inception_v3(pretrained=True)
            except:
                name = 'inception_v3_google'
                model_urls[name] = model_urls[name].replace('https://', 'http://')
                self.incept = torchvision.models.inception_v3(pretrained=True)
        else:
            self.incept = model
            
        self.incept.training = False
        self.incept.transform_input = False
        self.incept = self.incept.cuda()
        self.incept.eval()
    
        # init data transformer
        normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
           transforms.ToTensor(),
           normalize
        ])
        
        # init log
        try:
            self.log_fname = os.path.join("projects", 
                                          main.__file__, 
                                          "log", 
                                          "incept_score.txt")
        except:
            pass
        
    
    def _init_log(self):
        header = "incept_score"
        with open(self.log_fname, 'w') as f:
            f.write(header)
    
    def run(self):
        
        if os.path.isfile(self.log_fname) == False:
            self._init_log()
        
        # generate imgs
        X, Y, Z, Z_bar = self.sampler.next()
        imgs = np.zeros((0, X.shape[1], X.shape[2], X.shape[3]))
        while imgs.shape[0] < self.n_samples:
            X, Y, Z, Z_bar = self.sampler.next()

            if self.enc is not None:
                Z = ld_gan.utils.model_handler.apply_model(self.enc, X)
            if self.gen is not None:
                X = ld_gan.utils.model_handler.apply_model(self.gen, Z)
            imgs = np.concatenate([imgs, X])
        
        # get score
        score = self.score(imgs)[0]
        if self.real_data_score is not None:
            score = score / self.real_data_score
        
        # write log
        line = str(score)
        with open(self.log_fname, 'a') as f:
            f.write("\n" + line)
        
    
    def score(self, imgs, batch_size=32):
        """
        Function to compute the inception score
        
        Parameters
        ----------
        imgs : numpy array
            array of the shape (N, X, Y, C)
        batch_size : int
            batch size for the prediction with the inception net
        """
        
        print "compute inception score..."
        
        # preprocess images
        if imgs.shape[0] != 299 or imgs.shape[1] != 299:
            imgs = np.array([scipy.misc.imresize(img, (299, 299)) for img in imgs])
        n_batches = 1 + (len(imgs) / batch_size)
        batches = np.array_split(imgs, n_batches)
        
        # get prediction vectors of inception net for images
        preds = []
        for batch in tqdm(batches):
            imgs = [Image.fromarray(img) for img in batch]
            imgs = torch.stack([self.preprocess(img) for img in imgs])
            imgs = imgs.cuda()
            imgs = Variable(imgs)
            pred = self.incept(imgs)
            pred = F.softmax(pred)
            preds.append(pred.data.cpu().numpy())    
        preds = np.concatenate(preds)
        
        if len(preds) < 50000:
            print ">= 50000 samples needed, but only {} provided".format(len(preds))
        
        # compute inception score
        kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        score = np.exp(kl)
            
        return score
    
    
if __name__ == "__main__":
    
    trainset = torchvision.datasets.CIFAR10(root = '/tmp', download=True)
    x = trainset.train_data
    incept_score = InceptionScore()
    mean, std = incept_score.score(x)
    
    print "score = {} +- {}".format(mean, std)
    