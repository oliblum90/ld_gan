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
import ld_gan
import __main__ as main


class InceptionModeScore:
    
    def __init__(self, 
                 sampler=None, 
                 n_samples=None, 
                 enc=None, 
                 gen=None,
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
        try:
            self.incept = torchvision.models.inception_v3(pretrained=True)
        except:
            name = 'inception_v3_google'
            model_urls[name] = model_urls[name].replace('https://', 'http://')
            self.incept = torchvision.models.inception_v3(pretrained=True)
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
                                          "incept_mode_score.txt")
        except:
            pass
        
    
    def _init_log(self):
        header = "incept_mode_score"
        with open(self.log_fname, 'w') as f:
            f.write(header)
    
    def run(self):
        
        if os.path.isfile(self.log_fname) == False:
            self._init_log()
        
        # get imgs
        X, Y, Z, Z_bar = self.sampler.next()
        imgs_fake = np.zeros((0, X.shape[1], X.shape[2], X.shape[3]))
        imgs_real = []
        while imgs_fake.shape[0] < self.n_samples:
            X, Y, Z, Z_bar = self.sampler.next()
            imgs_real.append(X.copy())
            if self.enc is not None:
                Z = ld_gan.utils.model_handler.apply_model(self.enc, X)
            if self.gen is not None:
                X = ld_gan.utils.model_handler.apply_model(self.gen, Z)
            imgs_fake = np.concatenate([imgs_fake, X])
        imgs_real = np.concatenate(imgs_real)
        
        # get score
        score = self.score(imgs_fake, imgs_real)[0]
        if self.real_data_score is not None:
            score = score / self.real_data_score
        
        # write log
        line = str(score)
        with open(self.log_fname, 'a') as f:
            f.write("\n" + line)
        
    
    def score(self, imgs_fake, imgs_real, batch_size=32, splits=10):
        """
        Function to compute the inception score
        
        Parameters
        ----------
        imgs : numpy array
            array of the shape (N, X, Y, C)
        batch_size : int
            batch size for the prediction with the inception net
        splits : int
            The inception score is computed for a package of images.
            The variable 'splits' defines the number of these packages.
            Multiple computations of the score (for each package one) are 
            needed to compute a standard diviation (error) for the final
            score.
        """
        
        # preprocess fake images
        if imgs_fake.shape[0] != 299 or imgs_fake.shape[1] != 299:
            imgs_fake = [scipy.misc.imresize(img, (299, 299)) for img in imgs_fake]
            imgs_fake = np.array(imgs_fake)
        n_batches = 1 + (len(imgs_fake) / batch_size)
        batches = np.array_split(imgs_fake, n_batches)
        
        # get prediction vectors of inception net for fake imgs
        preds_fake = []
        for batch in batches:
            imgs = [Image.fromarray(img) for img in batch]
            imgs = torch.stack([self.preprocess(img) for img in imgs])
            imgs = imgs.cuda()
            imgs = Variable(imgs)
            pred = self.incept(imgs)
            pred = F.softmax(pred)
            preds_fake.append(pred.data.cpu().numpy())    
        preds_fake = np.concatenate(preds_fake)
        
        # preprocess real images
        if imgs_real.shape[0] != 299 or imgs_real.shape[1] != 299:
            imgs_real = [scipy.misc.imresize(img, (299, 299)) for img in imgs_real]
            imgs_real = np.array(imgs_real)
        n_batches = 1 + (len(imgs_real) / batch_size)
        batches = np.array_split(imgs_real, n_batches)
        
        # get prediction vectors of inception net for fake imgs
        preds_real = []
        for batch in batches:
            imgs = [Image.fromarray(img) for img in batch]
            imgs = torch.stack([self.preprocess(img) for img in imgs])
            imgs = imgs.cuda()
            imgs = Variable(imgs)
            pred = self.incept(imgs)
            pred = F.softmax(pred)
            preds_real.append(pred.data.cpu().numpy())    
        preds_real = np.concatenate(preds_real)
        
        # compute inception score
        scores = []
        for i in range(splits):
            part_fake = preds_fake[(i * preds_fake.shape[0] // splits): \
                         ((i + 1) * preds_fake.shape[0] // splits), :]
            part_real = preds_real[(i * preds_real.shape[0] // splits): \
                         ((i + 1) * preds_real.shape[0] // splits), :]

            p_star = np.expand_dims(np.mean(part_fake, 0), 0)
            p      = np.expand_dims(np.mean(part_real, 0), 0)

            kl = part_fake * (np.log(part_fake) - np.log(p))
            kl = np.mean(kl, 0)
            kl = kl - (p_star * (np.log(p_star) - np.log(p)))
            kl = np.exp(np.sum(kl))

            scores.append(kl)
            
        return np.mean(scores), np.std(scores)
    
    
if __name__ == "__main__":
    
    trainset = torchvision.datasets.CIFAR10(root = '/tmp', download=True)
    x = trainset.train_data
    incept_score = InceptionScore()
    mean, std = incept_score.score(x)
    
    print "score = {} +- {}".format(mean, std)
    