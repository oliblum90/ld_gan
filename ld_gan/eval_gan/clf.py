import os
from torch import nn
import torchvision
import torch.optim as optim
import ld_gan
import ld_gan.utils.utils as ld
from tqdm import tqdm
from sklearn.utils import shuffle
import numpy as np
import torch
from torch.autograd import Variable


def train_cnn(X, Y, 
              Xt, Yt,
              n_classes,
              batch_size = 32, 
              lr = 0.001, 
              n_epochs = 50, 
              save_model_step = 5, 
              save_model_path = ""):
    
    # load inception model
    cnn = torchvision.models.inception_v3(num_classes=n_classes, pretrained=False)
    cnn.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    opt = optim.Adam(cnn.parameters(), lr=lr)
    
    n_iters = int(len(X) / batch_size)
    
    fname = os.path.join(save_model_path, "acc.txt")
    with open(fname, 'w') as f:
            f.write("acc")
        
    for epoch in range(n_epochs):
        
        X, Y = shuffle(X, Y)
        
        for it in tqdm(range(n_iters)):
            
            # prepare data
            idx_lower = it * batch_size
            idx_upper = idx_lower + batch_size
            x_batch = X[idx_lower : idx_upper]
            y_batch = Y[idx_lower : idx_upper]
            x_batch, y_batch = ld_gan.data_proc.transform(x_batch, y_batch)
            
            # train clf
            cnn.zero_grad()
            y_pred, y_pred_aux = cnn(x_batch)
            err = criterion(y_pred, y_batch)
            err += criterion(y_pred_aux, y_batch)
            err.backward()
            opt.step()
            
        # get acc
        n_iters_test = int(len(Xt) / batch_size)
        acc = 0
        for it_test in range(n_iters_test):

            idx_lower = it_test * batch_size
            idx_upper = idx_lower + batch_size
            x_batch = Xt[idx_lower : idx_upper]
            y_batch = Yt[idx_lower : idx_upper]
            x_batch = ld_gan.data_proc.transform(x_batch)
            yt_pred = cnn(x_batch)[0]
            yt_pred = ld_gan.data_proc.transform(yt_pred)
            yt_pred = np.argmax(yt_pred, axis = 1)
            acc += float((yt_pred == y_batch).sum()) / len(y_batch)
        acc = acc / it_test
        
        fname = os.path.join(save_model_path, "acc.txt")
        with open(fname, 'a') as f:
                f.write("\n" + str(acc))
        
        # print status
        print "==========================="
        print "/n"
        print "EPOCH", epoch
        print "==========================="
        print "acc = ", acc

        # save model
        if epoch % save_model_step == 0:
            epoch_str = str(epoch).zfill(3)
            fname = os.path.join(save_model_path, "cnn_64_" + epoch_str + ".pth")
            torch.save(cnn, fname)



def train_cnn_load_live(path_train,
                        path_test,
                        batch_size = 32, 
                        lr = 0.001, 
                        n_epochs = 50, 
                        save_model_step = 5, 
                        save_model_path = "",
                        resize=128):
    
    n_classes = len([p for p in os.listdir(path_train) \
                     if os.path.isdir(os.path.join(path_train, p))])
    
    # load inception model
    cnn = torchvision.models.inception_v3(num_classes=n_classes, pretrained=False)
    cnn.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    opt = optim.Adam(cnn.parameters(), lr=lr)
    
    # prepare dataloader
    loader_train = ld_gan.data_proc.data_loader.live_loader(path_train, 
                                                            batch_size, 
                                                            resize=resize)
    loader_test = ld_gan.data_proc.data_loader.live_loader(path_test, 
                                                           batch_size, 
                                                           resize=resize)
    
    
    fname = os.path.join(save_model_path, "acc.txt")
    with open(fname, 'w') as f:
            f.write("acc")
        
    for epoch in range(n_epochs):
                
        for x_batch, y_batch in tqdm(loader_train):
            
            # train clf
            x_batch = Variable(x_batch.cuda()).float()
            y_batch = Variable(y_batch.cuda())
            cnn.zero_grad()
            y_pred, y_pred_aux = cnn(x_batch)
            err = criterion(y_pred, y_batch)
            err += criterion(y_pred_aux, y_batch)
            err.backward()
            opt.step()
            
        # get acc
        accs = []
        for x_batch, y_batch in tqdm(loader_test):
            y_pred, y_batch = ld_gan.data_proc.transform(y_pred, y_batch)
            y_pred = np.argmax(y_pred, axis=1)
            acc = (y_pred == y_batch).sum() / float(len(y_pred))
            accs.append(acc)
        fname = os.path.join(save_model_path, "acc.txt")
        acc = np.mean(accs)
        with open(fname, 'a') as f:
            f.write("\n" + str(acc))
        
        # print status
        print "==========================="
        print "/n"
        print "EPOCH", epoch
        print "==========================="
        print "acc = ", acc

        # save model
        if epoch % save_model_step == 0:
            epoch_str = str(epoch).zfill(3)
            fname = os.path.join(save_model_path, "cnn_" + epoch_str + ".pth")
            torch.save(cnn, fname)


