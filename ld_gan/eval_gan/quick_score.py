import ld_gan
import numpy as np
from ld_gan import visualize
from tqdm import tqdm

def i_score(project, epoch, X, sampler, n_samples=50000, norm=True):
    
    # load models
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    
    # prepare data
    x_fake = np.zeros((0,X.shape[1],X.shape[2],X.shape[3]))
    pbar = tqdm(total=n_samples)
    while len(x_fake) < n_samples:
        pbar.update(len(x_fake))
        _, _, Z, _ = sampler.next()
        imgs_fake = ld_gan.utils.model_handler.apply_model(gen, Z)
        x_fake = np.concatenate((x_fake, imgs_fake))
    pbar.close()
    x_fake = x_fake[:n_samples]
    score_fkt = ld_gan.eval_gan.InceptionScore()
    sorce_fake = score_fkt.score(x_fake)
    print "score_fake:", sorce_fake
    
    if norm:
        sorce_real = score_fkt.score(X)
        score = sorce_fake / sorce_real
        print "score_real:", sorce_real
        print "score     :", score
    else:
        score = sorce_fake
    
    return score



def im_score(project, epoch, X, sampler, n_samples=50000, norm=True):
    
    # load models
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    
    # prepare data
    x_fake = np.zeros((0,X.shape[1],X.shape[2],X.shape[3]))
    while len(x_fake) < n_samples:
        _, _, Z, _ = sampler.next()
        imgs_fake = ld_gan.utils.model_handler.apply_model(gen, Z)
        x_fake = np.concatenate((x_fake, imgs_fake))
    x_fake = x_fake[:n_samples]
    score_fkt = ld_gan.eval_gan.InceptionModeScore()
    sorce_fake = score_fkt.score(x_fake, X)
    print "score_fake:", sorce_fake
    
    if norm:
        sorce_real = score_fkt.score(X, X)
        score = sorce_fake / sorce_real
        print "score_real:", sorce_real
        print "score     :", score
    else:
        score = sorce_fake
    
    return score

