import os
os.chdir("../")

import torch.backends.cudnn as cudnn
import torch
import random
import ld_gan
import numpy as np
import sys


if __name__ == "__main__":
    
    gpu_id = int(sys.argv[1])
    
    with torch.cuda.device(gpu_id):

        RAND_SEED = 42
        cudnn.benchmark = True
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        torch.cuda.manual_seed_all(RAND_SEED)


        BATCH_SIZE  = 256
        LATENT_SIZE = 512
        LR_GAN  = 0.0002
        LR_AE   = 0.0001
        LR_FL   = 0.0001
        LR_CLF  = 0.0001
        LR_GCLF = 0.0001

        path = "data/faceScrub/imgs_top_aligned/"
        X, Y = ld_gan.data_proc.data_loader.load_data(path, resize=64, test_train="train")
        n_classes = Y.shape[1]
        Y = np.argmax(Y, axis = 1)
        img_size = X.shape[2]


        gen = ld_gan.models.gen.Gen(latent_size = LATENT_SIZE, ipt_size=img_size)
        dis = ld_gan.models.dis.Dis(ipt_size=img_size)
        enc = ld_gan.models.enc.Enc(n_features = LATENT_SIZE, ipt_size=img_size)
        gen.cuda()
        gen.apply(ld_gan.models.init_weights)
        dis.cuda()
        dis.apply(ld_gan.models.init_weights)
        enc.cuda()
        enc.apply(ld_gan.models.init_weights)


        clf_train_op = ld_gan.train_ops.Clf(enc, LR_CLF, LATENT_SIZE, n_classes)
        clf_layer = clf_train_op.clf_layer

        train_ops = [
                        ld_gan.train_ops.GanDis(gen, dis, LR_GAN),
                        ld_gan.train_ops.GanGen(gen, dis, LR_GAN),
                        ld_gan.train_ops.FLoss(enc, gen, LR_FL),
                        ld_gan.train_ops.VGGAutoEnc(enc, gen, LR_AE),
                        clf_train_op,
                        ld_gan.train_ops.GenCLF(gen, enc, clf_layer, Y, LR_GCLF)
                    ]


        sampler = ld_gan.sample.nn_sampler_scs(enc, X, Y, 
                                               BATCH_SIZE, 
                                               nn_search_radius = 50,
                                               n_neighbors = 5)


        trainer = ld_gan.trainer.Trainer(gen, 
                                         dis, 
                                         enc, 
                                         train_ops,
                                         sampler,
                                         len(X),
                                         gen_img_step      = 5,
                                         save_model_step   = 10,
                                         gen_iscore_step   = 10,
                                         gen_tsne_step     = 50,
                                         n_epochs          = 1000,
                                         batch_size        = BATCH_SIZE,
                                         gpu_id            = gpu_id)

        trainer.train()

