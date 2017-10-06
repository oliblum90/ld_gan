import os
import inspect, re
import numpy as np
# import cv2
import scipy.misc
import sklearn
from matplotlib import gridspec
import matplotlib
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from data_proc.transformer import np_to_tensor, tensor_to_np
import ld_gan
import datetime
import time
from tqdm import tqdm



def out(p):
    """
    function prints variable name and value. good for debugging
    """
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bout\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            print m.group(1), ":\t", p
            
            
def show_training_status(fname, epoch, d_loss, g_loss, d_test_loss):
    print "EPOCH: {}".format(epoch)
    print "--------------------------------------------------------"
    print "disc_loss: \t {} \t {}".format(d_loss, d_test_loss)
    print "gen_loss: \t {}".format(g_loss)
    print "--------------------------------------------------------"
    
    try:
        history = np.load(fname)
    except:
        history = {"d_loss"      : np.array([]),
                   "d_test_loss" : np.array([]),
                   "g_loss"      : np.array([])}
        history = np.array(history)
        
    history[()]["d_loss"] = np.append(history[()]["d_loss"], d_loss)
    history[()]["d_test_loss"] = np.append(history[()]["d_test_loss"], d_test_loss)
    history[()]["g_loss"] = np.append(history[()]["g_loss"], g_loss)
    
    np.save(fname, history)
    
            
def disp(img_list, title_list = None, fname = None, figsize=None):
    """
    display a list of images
    """
    import matplotlib.pylab as plt

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize, dpi=180)

    for idx, img in enumerate(img_list):

        plt.subplot(1, len(img_list), idx+1)
        if title_list is not None:
            plt.title(title_list[idx])
            
        img_show = img.copy()
        
        if len(img_show.shape) == 2:
            plt.imshow(img_show.astype(np.uint8), 
                       vmax = img.max(), vmin = img.min(),
                       cmap='gray')
        else:
            plt.imshow(img_show.astype(np.uint8), 
                       vmax = img.max(), vmin = img.min())
            
        plt.axis("off")
    
    if fname is not None:
        plt.savefig(fname)
        
    plt.show()
    
def disp_array(imgs, shape, fname = None, title = None):
    
    import matplotlib.pylab as plt
    
    plt.figure()
    if title is not None:
        plt.title(title)
    row = shape[0]
    col = shape[1]
    for idx, img in enumerate(imgs):
        if idx + 1 > row * col:
            break
        plt.subplot(row, col, idx + 1)
        plt.imshow(img.astype(np.uint8),vmax = img.max(), vmin = img.min())
        plt.axis("off")
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
    
def make_gif(imgs, gif_fname, fps = 10):
    
    # make temp img dir
    os.mkdir("tmp")
    
    # save images
    fnames = []
    for idx, sample in enumerate(imgs):
        plt.figure()
        plt.imshow(sample.astype(np.uint8))
        plt.axis("off")
        fname = os.path.join("tmp", str(idx) + ".png")
        fnames.append(fname)
        plt.savefig(fname)
    
    # make gif
    import imageio
    images = []
    for filename in fnames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_fname, images, fps = fps)
    
    # delete images
    import shutil
    shutil.rmtree('tmp')
    
def save_g_imgs(fname, 
                imgs, 
                imgs_per_row = 10, 
                return_img = False, 
                show_img = False):
        
    n_rows = len(imgs) / imgs_per_row
    rows = []
    for i in range(n_rows):
        lower_idx = i * 10
        upper_idx = lower_idx + 10
        rows.append(np.concatenate(imgs[lower_idx : upper_idx], axis = 1))
    img = np.concatenate(rows)
    
    # handle 1 channel images
    if img.shape[2] == 1:
        img = img[:,:,0]
    
    if return_img:
        return img
    elif show_img:
        disp([img])
    else:
        scipy.misc.imsave(fname, img)
    
#iters_per_epoch = 234
def learning_curve(project,
                   path            = "projects",
                   smooth          = 100,
                   show_disc       = True, 
                   show_gen_d      = True, 
                   show_gen_e      = True, 
                   show_gen_img    = False,
                   iters_per_epoch = 136.,
                   max_epoch       = None,
                   gen_img_epoch   = -1,
                   xmax            = None,
                   ymax            = None,
                   show_hist_tsne  = False):
    
    import matplotlib.pylab as plt
    
    fname = os.path.join(path, project, 'log/logs.txt')
    logs = np.loadtxt(fname)

    x = np.arange(logs.shape[0]) / iters_per_epoch
    xc = np.arange(smooth/2, logs.shape[0] - smooth/2 + 1) / iters_per_epoch

    plt.figure()
    
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1]) 
    plt.subplot(gs[0])
    
    ymax_temp = 0
    
    if show_disc:
        plt.plot(x, logs[:,0], c='g', alpha = 0.2)
        y = np.convolve(logs[:,0], 
                        np.ones((smooth,))/smooth, 
                        mode='valid')
        l1 = plt.plot(xc, y, c='g', label='loss_disc')
        ymax_temp = max(ymax_temp, y.max())

    if show_gen_d:
        plt.plot(x, logs[:,1], c='b', alpha = 0.2)
        y = np.convolve(logs[:,1], 
                        np.ones((smooth,))/smooth, 
                        mode='valid')
        l2 = plt.plot(xc, y, c='b', label='loss_gen_d')
        ymax_temp = max(ymax_temp, y.max())

    if show_gen_e:
        plt.plot(x, logs[:,2], c='r', alpha = 0.2)
        y = np.convolve(logs[:,2], 
                        np.ones((smooth,))/smooth, 
                        mode='valid')
        l3 = plt.plot(xc, y, c='r', label='loss_gen_e')
        ymax_temp = max(ymax_temp, y.max())
        
    if show_gen_img:
        plt.plot(x, logs[:,3], c='r', alpha = 0.2)
        y = np.convolve(logs[:,3], 
                        np.ones((smooth,))/smooth, 
                        mode='valid')
        l4 = plt.plot(xc, y, c='gold', label='loss_gen_img')
        ymax_temp = max(ymax_temp, y.max())

    # adjust the plot range
    if ymax is None:
        ymax = ymax_temp
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    if xmax is not None:
        axes.set_xlim([0,xmax])
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(fontsize = 8)

    
    path_img = os.path.join(path, project, 'generated_img')
    if gen_img_epoch == -1:
        gen_img_epoch = int(os.listdir(path_img)[-1][:4])
    
    # show fake image
    plt.subplot(gs[1])
    
    fname = str(gen_img_epoch).zfill(4) + "_fake.png"
    fname = os.path.join(path_img, fname)
    try:
        img = scipy.misc.imread(fname)
    except:
        print "only found the files {}".format(os.listdir(path_img))
    
    plt.imshow(img[0:img.shape[1]], cmap='gray')
    plt.title("fake imgs (epoch {})".format(gen_img_epoch))
    plt.axis('off')
    
    # show real image
    plt.subplot(gs[2])
    
    fname = str(gen_img_epoch).zfill(4) + "_real.png"
    fname = os.path.join(path_img, fname)
    try:
        img = scipy.misc.imread(fname)
    except:
        print "only found the files {}".format(os.listdir(path_img))
    
    plt.imshow(img[0:img.shape[1]], cmap='gray')
    plt.title("real imgs")
    plt.axis('off')
    
    plt.show()
    
    path_ht = os.path.join(path, project, 'hist_tsne')
    if show_hist_tsne:
        try:
            fname = os.path.join(path_ht, project, '0_hist_tsne.png')
            img = scipy.misc.imread(fname)
            disp([img])
        except:
            fname = str(gen_img_epoch).zfill(4) + "_hist_tsne.png"
            fname = os.path.join(path_ht, fname)
            img = scipy.misc.imread(fname)
            disp([img])
     
    
    
def learning_curve_ia(project,
                      path            = "projects",
                      smooth          = 100,
                      iters_per_epoch = 136.,
                      max_epoch       = None,
                      xmax            = None,
                      ymax            = None,
                      show_hist_tsne  = False,
                      high_res        = False,
                      logs_fname      = None,
                      mean            = True):
    
    import matplotlib.pylab as plt
    matplotlib.rcParams.update({'font.size': 8})
    
    if mean:
        img_ending = "_fake.png"
    else:
        img_ending = "_mean_fake.png"
    
    try:
        fname = "projects/" + project + "/log/iters_per_epoch"
        iters_per_epoch = float(np.loadtxt(fname))
    except:
        print "no iters_per_epoch file"
    
    if logs_fname is None:
        fname = os.path.join(path, project, 'log/logs.txt')
    else:
        fname = os.path.join(path, project, 'log', logs_fname)
        logs = np.loadtxt(fname, skiprows=1, delimiter=" ")
        fname_tmp = os.path.join(path, project, 'log/logs.txt')
        logs_tmp = np.loadtxt(fname_tmp, skiprows=1, delimiter=" ")
        iters_per_epoch = (len(logs) / float(len(logs_tmp))) * iters_per_epoch        
        
    logs = np.loadtxt(fname, skiprows=1, delimiter=" ")
    labels = np.genfromtxt(fname, dtype='str')
    labels = labels[0] if labels.ndim > 1 else labels
    colors = ['g', 'b', 'r', 'gold', 'k', 'magenta', 'lime', 'cyan'][:len(labels)]
    
    try:
        for i in range(10):
            for i1 in range(logs.shape[0]):
                for i2 in range(logs.shape[1]):
                    logs[i1, i2] = logs[i1, i2] if logs[i1, i2]!=-1000 else logs[i1, i2-1]
    except:
        pass

    x = np.arange(logs.shape[0]) / iters_per_epoch
    xc = np.arange(smooth/2, logs.shape[0] - smooth/2 + 1) / iters_per_epoch
    if high_res:
        if show_hist_tsne:
            fig = plt.figure(figsize=(7.5,5), dpi=180)
        else:
            fig = plt.figure(figsize=(7.5,2.5), dpi=180)
    else:
        if show_hist_tsne:
            fig = plt.figure(figsize=(7.5,5))
        else:
            fig = plt.figure(figsize=(7.5,2))
    
    if show_hist_tsne:
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1]) 
    else:
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1]) 
        
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    if show_hist_tsne:
        ax4 = fig.add_subplot(gs[3:])

    ax2.axis('off')
    ax3.axis('off')
    if show_hist_tsne:
        ax4.axis('off')

    ymax_temp = 0
    
    if logs.ndim == 1:
        logs = np.expand_dims(logs, axis = 1)
        
    for idx in range(logs.shape[1]):
        if smooth != 0:
            ax1.plot(x, logs[:,idx], c=colors[idx], alpha = 0.2)
            y = np.convolve(logs[:,idx], 
                            np.ones((smooth,))/smooth, 
                            mode='valid')
            l1 = ax1.plot(xc, y, c=colors[idx], label=labels[idx])
            ymax_temp = max(ymax_temp, y.max())
        else:
            ax1.plot(x, logs[:,idx], c=colors[idx])
            ymax_temp = max(ymax_temp, logs[:,idx].max())

    # adjust the plot range
    if ymax is None:
        ymax = ymax_temp
    ax1.set_ylim([0,ymax])
    if xmax is not None:
        ax1.set_xlim([0,xmax])
        
    #ax1.xlabel('epoch')
    #ax1.ylabel('loss')
    ax1.legend(fontsize = 8)

    
    path_img = os.path.join(path, project, 'generated_img')
    path_ht = os.path.join(path, project, 'hist_tsne')
    gen_img_epoch = int(os.listdir(path_img)[-1][:4])
    try:
        gen_tsne_epoch = int(os.listdir(path_ht)[-1][:4])
    except:
        gen_tsne_epoch = 0
    
    # show fake image
    fname = str(gen_img_epoch).zfill(4) + img_ending
    fname = os.path.join(path_img, fname)
    try:
        img = scipy.misc.imread(fname)
    except:
        print "only found the files {}".format(os.listdir(path_img))
    
    ax2.imshow(img[0:img.shape[1]], cmap='gray')
    ax2.set_title("fake imgs (epoch {})".format(gen_img_epoch))
    ax2.axis('off')
    
    # show real image
    
    fname = str(gen_img_epoch).zfill(4) + "_real.png"
    fname = os.path.join(path_img, fname)
    try:
        img = scipy.misc.imread(fname)
    except:
        print "only found the files {}".format(os.listdir(path_img))
    
    ax3.imshow(img[0:img.shape[1]], cmap='gray')
    ax3.set_title("real imgs")
    ax3.axis('off')
        
    if show_hist_tsne:
        try:
            fname = os.path.join(path_ht, '0_hist_tsne.png')
            img = scipy.misc.imread(fname)
            ax4.imshow(img)
            ax4.set_title("hist / tsne")
        except:
            fname = str(gen_tsne_epoch).zfill(4) + "_hist_tsne.png"
            fname = os.path.join(path_ht, fname)
            img = scipy.misc.imread(fname)
            ax4.imshow(img)
            ax4.set_title("hist / tsne (epoch {})".format(gen_tsne_epoch))
    
    def onclick(event):
        
        imgs = os.listdir(path_img)
        iters = [int(img_str[:4]) for img_str in imgs]
        idx = np.argmin(np.abs(np.array(iters) - event.xdata))
        gen_img_epoch = iters[idx]
                
        fname = str(gen_img_epoch).zfill(4) + img_ending
        fname = os.path.join(path_img, fname)
        img = scipy.misc.imread(fname)
        ax2.imshow(img[0:img.shape[1]], cmap='gray')
        ax2.set_title("fake imgs (epoch {})".format(gen_img_epoch))
                
        fname = str(gen_img_epoch).zfill(4) + "_real.png"
        fname = os.path.join(path_img, fname)
        img = scipy.misc.imread(fname)
        ax3.imshow(img[0:img.shape[1]], cmap='gray')
        
        if show_hist_tsne:
            try:
                fname = os.path.join(path_ht, '0_hist_tsne.png')
                img = scipy.misc.imread(fname)
                ax4.imshow(img)
                ax4.set_title("hist / tsne")
            except:
                hts = os.listdir(path_ht)
                iters = [int(ht_str[:4]) for ht_str in hts]
                idx = np.argmin(np.abs(np.array(iters) - event.xdata))
                gen_tsne_epoch = iters[idx]
                fname = str(gen_tsne_epoch).zfill(4) + "_hist_tsne.png"
                fname = os.path.join(path_ht, fname)
                img = scipy.misc.imread(fname)
                ax4.imshow(img)
                ax4.set_title("hist / tsne (epoch {})".format(gen_tsne_epoch))
                
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    
def plot_hist_and_tsne(f_X, 
                       y = None, 
                       imgs = None,
                       fname = None, 
                       n_pts_tsne = 1000,
                       n_bins = 50,
                       n_clusters = 30,
                       project = None,
                       epoch_str = None):
    
    if fname is not None:
        matplotlib.use('Agg')
        import matplotlib.pylab as plt
    else:
        import matplotlib.pylab as plt
        
    plt.close('all')
    
    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # hist
    ax1.set_title("histogram")
    ax1.hist(f_X.flatten(), bins = n_bins)
    
    # tsne
    ax2.set_title("t-SNE")
    tsne = TSNE(n_components=2, 
                random_state=0, 
                metric = 'cosine', 
                learning_rate=1000)
    X_tsne = tsne.fit_transform(f_X[:n_pts_tsne])
    
    fname_pts = project + "/tsne_pts/" \
                + epoch_str + "_" + str(n_pts_tsne) + ".npy"
    try:
        os.mkdir(project + "/tsne_pts")
    except:
        pass
    np.save(fname_pts, X_tsne)
    
    if y is None:
        ax2.scatter(X_tsne[:,0], X_tsne[:,1], s = 10, alpha = 0.3)
    else:
        ax2.scatter(X_tsne[:,0], X_tsne[:,1], c=y[:n_pts_tsne], s = 10, alpha = 0.3)
        
    if imgs is not None:
        from sklearn.cluster import KMeans
        from matplotlib import offsetbox
        kmeans = KMeans(n_clusters=n_clusters).fit(X_tsne)
        kmeans = kmeans.cluster_centers_
        nbrs = NearestNeighbors(n_neighbors = 1).fit(X_tsne)
        dists, idxs = nbrs.kneighbors(kmeans)
        
        for mean, img in zip(kmeans, imgs[idxs]):
            img = np.squeeze(img)
            zoom = 0.7 / (img.shape[0] / 32)
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, 
                                                                      cmap=plt.cm.gray,
                                                                      zoom = zoom),
                                                 mean,
                                                 frameon=False)
            ax2.add_artist(imagebox)

    
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

        
def gen_real_compare(weights_gen, 
                     weights_enc, 
                     imgs_real,
                     enc_norm = 'sig2',
                     n_col_channels = 1,
                     n_gan_layers = 3,
                     latent_size = 256,
                     filter_mult = 64,
                     return_imgs = False):

    from ls_gan import models
    
    n_imgs = len(imgs_real)
    
    enc = models.enc(weights_fname = weights_enc, 
                 include_pred_layer = False,
                 inclue_feature_relu=False,
                 normalize_feature_layer=enc_norm,
                 n_col_channels = n_col_channels)

    generator = models.gen.one_class_gen(n_upsampling_layers=n_gan_layers, 
                                         latent_size = latent_size,
                                         filter_multiplier=filter_mult,
                                         n_col_channels=n_col_channels)
    
    generator.load_weights(weights_gen)
    
    z_enc = enc.predict(imgs_real, batch_size=100)
    
    imgs_gen = generator.predict(z_enc[:n_imgs], batch_size=100)
    
    imgs_gen  = save_g_imgs("", imgs_gen, return_img = True)
    imgs_real = save_g_imgs("", imgs_real, return_img = True)
    
    if return_imgs:
        return imgs_gen, imgs_real
    else:
        disp([imgs_gen, imgs_real], title_list = ["imgs_fake", "imgs_real"])

        
def tsne_real_fake_vis(imgs_real, 
                       y, 
                       project, 
                       epoch,
                       z_mapped = None, 
                       sampler = None,
                       n_pts_tsne = 4000,
                       n_neighbors = 5,
                       alpha = 0.003,
                       real_img_mode = "single",
                       is_parallel_model = False,
                       reload_tsne       = False):
    
    import matplotlib.pylab as plt
    
    # load models
    if is_parallel_model:
        enc = ld_gan.utils.model_handler.load_parallel_model(project, epoch, "enc")
        gen = ld_gan.utils.model_handler.load_parallel_model(project, epoch, "gen")
    else:
        enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    
    if z_mapped is None:
        epoch_str = str(epoch).zfill(4)
        fname = "projects/" + project + "/tsne_pts/" \
                          + epoch_str + "_" + str(n_pts_tsne) + ".npy"
        
        try:
            if reload_tsne:
                jjj
            z_mapped = np.load(fname)
        
        except:
            print "compute tsne..."
            f_X = ld_gan.utils.model_handler.apply_model(enc, 
                                                         imgs_real[:n_pts_tsne],
                                                         1000)
            tsne = TSNE(n_components=2, 
                        random_state=0, 
                        metric = 'cosine', 
                        learning_rate=1000)
            z_mapped = tsne.fit_transform(f_X)
            try:
                os.mkdir("projects/" + project + "/tsne_pts")
            except:
                pass
            np.save(fname, z_mapped)
                         
            
        
        
    nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(z_mapped)
    
    fig = plt.figure(figsize = (7.5,2.5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    
    ax2.axis('off')
    ax3.axis('off')

    ax1.scatter(z_mapped[:, 0], z_mapped[:, 1], c = y[:len(z_mapped)], 
                s = 10, alpha = alpha, edgecolors='none')
    
    
    dists, idxs = nbrs.kneighbors(np.array([[0, 0]]))
    imgs = imgs_real[idxs[0]]
    z_enc = ld_gan.utils.model_handler.apply_model(enc, imgs_real[idxs[0]])
    
    
    
    
    def onclick(event):
        dists, idxs = nbrs.kneighbors(np.array([[event.xdata, event.ydata]]))
        idx = idxs[0][0]
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, idx, event.y, event.xdata, event.ydata))
        
        if real_img_mode == "all":
            imgs = imgs_real[idxs[0]]
            img = save_g_imgs(None, 
                              imgs, 
                              return_img=True,
                              imgs_per_row=int(np.ceil(np.sqrt(3))))
            ax2.imshow(img, cmap='gray')
            ax2.set_title(idxs[0])
            
        elif real_img_mode == "mean":
            imgs = imgs_real[idxs[0]]
            img = np.mean(imgs, axis = 0)
            img = img.astype(np.uint8)
            ax2.imshow(img, cmap='gray')
            ax2.set_title(idxs[0])
            
        elif real_img_mode == "all+mean":
            imgs = imgs_real[idxs[0]]
            imgs_small = np.concatenate(imgs, axis=1)
            imgs_small = scipy.misc.imresize(imgs_small, (imgs[0].shape[0] / len(imgs), 
                                                          imgs[0].shape[0]))
            img_mean = np.mean(imgs, axis = 0)
            img_mean = img_mean.astype(np.uint8)
            img = np.concatenate([imgs_small, img_mean], axis=0)
            ax2.imshow(img, cmap='gray')
            ax2.set_title(idxs[0])
            
        elif real_img_mode == "single":
            ax2.set_title("real img")
            img_real = np.squeeze(imgs_real[idx])
            ax2.imshow(img_real, cmap='gray')

        z_enc = ld_gan.utils.model_handler.apply_model(enc, imgs_real[idxs[0]])
        z_enc = np.mean(np.array(z_enc), axis = 0)
        
        if sampler is not None:
            z_enc = sampler(z_enc)
        
        if z_enc.ndim == 1:
            z_enc = np.array([z_enc])
        
        img_fake = ld_gan.utils.model_handler.apply_model(gen, z_enc)
                
        ax3.set_title("fake img")
        img_fake = np.squeeze(img_fake)
        ax3.imshow(img_fake, cmap='gray')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
    
    
    
    
    
def tsne_to_interpol_arr(imgs_real, 
                         y, 
                         project, 
                         epoch,
                         small = False,
                         n_imgs = 5,
                         all_fake = False,
                         z_mapped = None, 
                         sampler = None,
                         n_pts_tsne = 4000,
                         n_neighbors = 5,
                         alpha = 0.003,
                         real_img_mode = "single",
                         recompute_tsne = False):
    
    import matplotlib.pylab as plt
    
    enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    
    f_X = ld_gan.utils.model_handler.apply_model(enc, imgs_real, 100)
    
    if z_mapped is None:
        epoch_str = str(epoch).zfill(4)
        fname = "projects/" + project + "/tsne_pts/" \
                          + epoch_str + "_" + str(n_pts_tsne) + ".npy"
        
        try:
            if recompute_tsne:
                jjj
            z_mapped = np.load(fname)
        
        except:
            print "compute tsne..."
            tsne = TSNE(n_components=2, 
                        random_state=0, 
                        metric = 'cosine', 
                        learning_rate=1000)
            z_mapped = tsne.fit_transform(f_X[:n_pts_tsne])
            try:
                os.mkdir("projects/" + project + "/tsne_pts")
            except:
                pass
            np.save(fname, z_mapped)
                         
    
    if small:
        fig = plt.figure(figsize = (6.,3.))
    else:
        fig = plt.figure(figsize = (10.,5.))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    ax2.axis('off')

    ax1.scatter(z_mapped[:, 0], z_mapped[:, 1], c = y[:len(z_mapped)], 
                s = 10, alpha = alpha, edgecolors='none')
    
    
    arr_img = None 
    
    np.save("/tmp/z_mapped.npy", z_mapped)
    np.save("/tmp/f_X.npy", f_X)
    
    
    def onclick(event):
        
        global arr_img
        
        z_mapped = np.load("/tmp/z_mapped.npy")
        
        if (event.xdata > 200) or (event.ydata > 200):
            path = os.path.join("projects/", 
                                project,
                                "_demo_imgs")
            if os.path.isdir(path) == False:
                os.mkdir(path)
            path = os.path.join(path, str(epoch))
            if os.path.isdir(path) == False:
                os.mkdir(path)
            idx = len(os.listdir(path))
            fname = os.path.join(path, str(idx) + ".png")
            scipy.misc.imsave(fname, arr_img)
            ax2.set_title("saved img {}".format(fname))
            return
        
        ax2.set_title("")
        
        click_pos = np.array([[event.xdata, event.ydata]])
        dists = pairwise_distances(click_pos, z_mapped)
        idx = np.argsort(dists, axis=1)[:, 0]
        z_pos = f_X[idx]
        dists = pairwise_distances(z_pos, f_X, metric='cosine')
        idxs = np.argsort(dists, axis=1)[:, :n_neighbors]
                         
        imgs = imgs_real[idxs[0]]
        z_encs = ld_gan.utils.model_handler.apply_model(enc,imgs_real[idxs[0]])
                
        Z_enc_00 = z_encs[0]
        Z_enc_10 = z_encs[1]
        Z_enc_01 = z_encs[2]
        Z_enc_11 = z_encs[3]
                
        img_size = imgs_real.shape[1]
        arr_img = np.zeros((img_size * n_imgs, img_size * n_imgs, 3))
                
        for x in range(n_imgs):
            for y in range(n_imgs):
                                
                x_factor = x / float(n_imgs-1.)
                y_factor = y / float(n_imgs-1.)
                
                
                z_00_factor = (1. - x_factor) * (1. - y_factor)
                z_10_factor = (x_factor) * (1. - y_factor)
                z_01_factor = (1. - x_factor) * (y_factor)
                z_11_factor = (x_factor) * (y_factor)
                                
                z_enc = Z_enc_00 * z_00_factor + \
                        Z_enc_10 * z_10_factor + \
                        Z_enc_01 * z_01_factor + \
                        Z_enc_11 * z_11_factor
                
                z_enc = np.array([z_enc])
                        
                if sampler is not None:
                    z_enc = sampler(z_enc)
                
                img_fake = ld_gan.utils.model_handler.apply_model(gen, z_enc)
                        
                pos_x_min = x * img_size
                pos_x_max = (x+1) * img_size
                pos_y_min = y * img_size
                pos_y_max = (y+1) * img_size
                
                arr_img[pos_x_min:pos_x_max, pos_y_min:pos_y_max] = img_fake
                
        if all_fake == False:
            arr_img[:img_size, :img_size] = imgs[0]
            arr_img[:img_size, -img_size:] = imgs[2]
            arr_img[-img_size:, :img_size] = imgs[1]
            arr_img[-img_size:, -img_size:] = imgs[3]
        
        arr_img = arr_img.astype('uint8')
        
        ax2.imshow(arr_img, cmap='gray')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
    
    
def discriminator_certanty_curve(imgs_real, 
                                 y,
                                 project, 
                                 epoch,
                                 z_mapped = None,
                                 n_pts_tsne = 4000,
                                 alpha = 0.3,
                                 batch_size = 512,
                                 n_interpol_imgs = 13,
                                 dis_score_mode='center',
                                 loss = None):
    
    ymin, ymax = 0., 0.5
    
    import matplotlib.pylab as plt
    
    # load models
    enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    dis = ld_gan.utils.model_handler.load_model(project, epoch, "dis")
   
    # load tsne 
    if z_mapped is None:
        epoch_str = str(epoch).zfill(4)
        fname = "projects/" + project + "/tsne_pts/" \
                          + epoch_str + "_" + str(n_pts_tsne) + ".npy"
        try:
            z_mapped = np.load(fname)
        except:
            print "compute tsne..."
            f_X = ld_gan.utils.model_handler.apply_model(enc, imgs_real[:n_pts_tsne])
            tsne = TSNE(n_components=2, 
                        random_state=0, 
                        metric = 'cosine', 
                        learning_rate=1000)
            z_mapped = tsne.fit_transform(f_X)
            try:
                os.mkdir("projects/" + project + "/tsne_pts")
            except:
                pass
            np.save(fname, z_mapped)
    nbrs = NearestNeighbors(n_neighbors = 1).fit(z_mapped)

    # load fig
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize = (10.,10.))
    gs = gridspec.GridSpec(4, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax21 = fig.add_subplot(gs[0, 1])
    ax22 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])
    ax21.axis('off')
    ax22.axis('off')
    
    gs00 = gridspec.GridSpecFromSubplotSpec(1, n_interpol_imgs, subplot_spec=gs[3, :])
    interpolate_img_axs = [fig.add_subplot(g) for g in gs00]
    [ax.axis('off') for ax in interpolate_img_axs]
    
    # plot tsne
    ax1.scatter(z_mapped[:, 0], z_mapped[:, 1], c = y[:len(z_mapped)], 
                s = 10, alpha = alpha, edgecolors='none')
    
    z_encs = ld_gan.utils.model_handler.apply_model(enc, 
                                                    imgs_real, 
                                                    batch_size = batch_size)
    
    np.save('/tmp/click.npy', np.array([0]))
    
    def onclick(event):
        
        dists, idxs = nbrs.kneighbors(np.array([[event.xdata, event.ydata]]))
        idx = idxs[0][0]
        
        click = np.load('/tmp/click.npy')[0]
        
        if '0.125,0.71' in str(event.inaxes):
            
            # plot chosen img
            if click == 0:
                
                _, idxs = nbrs.kneighbors(np.array([[event.xdata, event.ydata]]))
                img = imgs_real[idxs[0][0]]
                ax21.imshow(img, cmap='gray')
                x = np.array([img])
                d = ld_gan.utils.model_handler.apply_model(dis, x)
                ax21.set_title("dis(img) = " + str(np.round(d, 3)))

                # create ordered dis curve
                z = ld_gan.utils.model_handler.apply_model(enc, x)
                z_current = z.copy()
                z = np.tile(z, (len(z_encs), 1))
                
                if dis_score_mode == 'center':
                    z = (z_encs + z) / 2.0
                    x = ld_gan.utils.model_handler.apply_model(gen, z,
                                                               batch_size=batch_size)
                    d = ld_gan.utils.model_handler.apply_model(dis, x,
                                                               batch_size=batch_size)
                
                elif dis_score_mode == 'mean':
                    factors = np.linspace(0, 1, 10)
                    zs = [z_encs*f + z*(1-f) for f in factors]
                    dis_scores = []
                    for z in zs:
                        x = ld_gan.utils.model_handler.apply_model(gen, z,
                                                                   batch_size=batch_size)
                        d = ld_gan.utils.model_handler.apply_model(dis, x,
                                                                   batch_size=batch_size)
                        dis_scores.append(d)
                    d = np.mean(np.array(dis_scores), axis=0)
                
                elif dis_score_mode == 'min':
                    factors = np.linspace(0, 1, 10)
                    zs = [z_encs*f + z*(1-f) for f in factors]
                    dis_scores = []
                    for z in zs:
                        x = ld_gan.utils.model_handler.apply_model(gen, z,
                                                                   batch_size=batch_size)
                        d = ld_gan.utils.model_handler.apply_model(dis, x,
                                                                   batch_size=batch_size)
                        dis_scores.append(d)
                    d = np.min(np.array(dis_scores), axis=0)
                
                if loss == "bce":
                    d = -np.log(d)                    
                    
                idxs_sorted = np.argsort(d)
                d_sorted = d[idxs_sorted]
                imgs_real_sorted = imgs_real[idxs_sorted]
                z_sorted = z[idxs_sorted]
                ax3.clear()
                ax3.text(0.01, 0.9,'DisScore-Img-Plot', ha='left', va='center',
                         transform=ax3.transAxes)
                ax3.plot(range(len(d)), d_sorted)
                ax3.set_ylim([ymin, ymax])

                np.save('/tmp/idxs_sorted.npy', idxs_sorted)
                np.save('/tmp/z_current.npy', z_current)
                np.save('/tmp/z_sorted.npy', z_sorted)
                np.save('/tmp/d_sorted.npy', d_sorted)
                np.save('/tmp/imgs_real_sorted.npy', imgs_real_sorted)
                np.save('/tmp/d.npy', d)

                ax1.clear()
                ax1.scatter(z_mapped[:, 0], z_mapped[:, 1], c = d[:len(z_mapped)], 
                            s = 10, alpha = alpha, edgecolors='none',
                            vmin=ymin, vmax=ymax)

                ax22.clear()
                ax22.axis('off')
                ax4.clear()
                ax4.text(0.01, 0.9,'DisScore-LatentPath-Plot', ha='left', va='center',
                     transform=ax4.transAxes)
                [ax.clear() for ax in interpolate_img_axs]
                [ax.axis('off') for ax in interpolate_img_axs]
                
                np.save('/tmp/click.npy', np.array([1]))
            
            if click == 1:
                _, idxs = nbrs.kneighbors(np.array([[event.xdata, event.ydata]]))
                img = imgs_real[idxs[0][0]]
                
                idx = int(event.xdata)
                z_sorted = np.load('/tmp/z_sorted.npy')
                imgs_real_sorted = np.load('/tmp/imgs_real_sorted.npy')
                z1 = np.load('/tmp/z_current.npy')

                d = ld_gan.utils.model_handler.apply_model(dis, np.array([img]))
                ax22.set_title("dis(img) = " + str(np.round(d, 3)))
                ax22.imshow(img)

                z2 = ld_gan.utils.model_handler.apply_model(enc, np.array([img]))            
                factors = np.linspace(0, 1, 100)
                z_interpolate = np.array([z2*f + z1*(1-f) for f in factors])
                x_interpolate = ld_gan.utils.model_handler.apply_model(gen, z_interpolate)
                d_interpolate = ld_gan.utils.model_handler.apply_model(dis, x_interpolate)
                ax4.clear()
                ax4.text(0.01, 0.9,'DisScore-LatentPath-Plot', ha='left', va='center',
                         transform=ax4.transAxes)
                ax4.plot(range(len(d_interpolate)), d_interpolate)
                ax4.set_ylim([ymin, ymax])
                idxs_ip = np.linspace(0, 99, n_interpol_imgs).astype('int')

                for idx, ax in zip(idxs_ip, interpolate_img_axs):
                    ax.imshow(x_interpolate[idx])
                    
                idxs_sorted = np.load('/tmp/idxs_sorted.npy')
                d_sorted = np.load('/tmp/d_sorted.npy')
                d = np.load('/tmp/d.npy')
                x = np.where(d_sorted == d[idxs[0][0]])[0][0]
                y = d[idxs[0][0]]
                ax3.clear()
                ax3.text(0.01, 0.9,'DisScore-Img-Plot', ha='left', va='center',
                         transform=ax3.transAxes)
                ax3.plot(range(len(d_sorted)), d_sorted)
                ax3.set_ylim([ymin, ymax])
                ax3.scatter([x], [y])

                #np.save('/tmp/click.npy', np.array([0]))
                
            
            
        if '(0.547727' in str(event.inaxes):
            ax1.set_title('ax2')
            
        if '(0.125,0.51' in str(event.inaxes):

            idx = int(event.xdata)
            z_sorted = np.load('/tmp/z_sorted.npy')
            imgs_real_sorted = np.load('/tmp/imgs_real_sorted.npy')
            z1 = np.load('/tmp/z_current.npy')
            
            img = imgs_real_sorted[idx]
            d = ld_gan.utils.model_handler.apply_model(dis, np.array([img]))
            ax22.set_title("dis(img) = " + str(np.round(d, 3)))
            ax22.imshow(img)
            
            z2 = ld_gan.utils.model_handler.apply_model(enc, np.array([img]))            
            factors = np.linspace(0, 1, 100)
            z_interpolate = np.array([z2*f + z1*(1-f) for f in factors])
            x_interpolate = ld_gan.utils.model_handler.apply_model(gen, z_interpolate)
            d_interpolate = ld_gan.utils.model_handler.apply_model(dis, x_interpolate)
            ax4.clear()
            ax4.text(0.01, 0.9,'DisScore-LatentPath-Plot', ha='left', va='center',
                     transform=ax4.transAxes)
            ax4.plot(range(len(d_interpolate)), d_interpolate)
            
            idxs_sorted = np.load('/tmp/idxs_sorted.npy')
            d_sorted = np.load('/tmp/d_sorted.npy')
            x = idx
            y = d_sorted[x]
            ax3.clear()
            ax3.text(0.01, 0.9,'DisScore-Img-Plot', ha='left', va='center',
                     transform=ax3.transAxes)
            ax3.plot(range(len(d_sorted)), d_sorted)
            ax3.set_ylim([ymin, ymax])
            ax3.scatter([x], [y])
            
            ax4.set_ylim([ymin, ymax])
            idxs = np.linspace(0, 100-1, n_interpol_imgs).astype('int')
            
            for idx, ax in zip(idxs, interpolate_img_axs):
                ax.imshow(x_interpolate[idx])
                
            ax1.set_title([ymin, ymax])
            
        if '(0.125,0.31' in str(event.inaxes):
            ax1.set_title('ax4')
            
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
    
def move_in_z_space(project, img1, img2, epoch, sampler = None):
    
    import matplotlib.pylab as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    
    # load models
    enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    
    # compute z-start and z-end
    z0, z1 = ld_gan.utils.model_handler.apply_model(enc, np.array([img1, img2]))
    z0 = z0 if sampler is None else sampler(z0)
    z1 = z1 if sampler is None else sampler(z1)
    
    # plot
    axis_color = 'lightgoldenrodyellow'
    fig = plt.figure()

    # Draw the plot
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)

    idx = 0

    img = ld_gan.utils.model_handler.apply_model(gen, np.array([z0]))
    pic = ax.imshow(img, cmap = 'gray')
    ax.axis('off')

    slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axis_color)
    slider = Slider(slider_ax, 'z', 0.0, 1.0, valinit=0.0)

    def sliders_on_changed(val):
        z = z0 + (z1 - z0) * slider.val
        img = ld_gan.utils.model_handler.apply_model(gen, np.array([z]))
        pic.set_data(img)
        # pic.set_data(X[int(slider.val),:,:,0])
        fig.canvas.draw_idle()
    slider.on_changed(sliders_on_changed)

    plt.show()
    


    
def show_base_and_neighbors(project, imgs, base_img_idx=None, n_neighbors=5):
    
    path = os.path.join("projects", project, "model")
    epochs = sorted([int(g[2:6]) for g in os.listdir(path) if "g_" in g])
    
    if base_img_idx is None:
        base_img_idx = np.random.randint(0, len(imgs)-1, 1)
    else:
        base_img_idx = [base_img_idx]
        
    base_img = imgs[base_img_idx]
    disp(base_img, figsize=(6,2))
    
    for epoch in epochs:
        
        enc = ld_gan.utils.model_handler.load_model(project, epoch, 'enc')
        z_all = ld_gan.utils.model_handler.apply_model(enc, imgs, 1000)
        z_base = z_all[base_img_idx]

        dists = sklearn.metrics.pairwise_distances(z_all, z_base)
        dists = np.squeeze(dists)
        idxs = np.argsort(dists)[1:n_neighbors+1]
        
        nn_imgs = imgs[idxs]
        
        disp(nn_imgs, figsize=(6,2))
    
    
def time_eval(project):
    
    import matplotlib.pylab as plt
    
    fname = os.path.join("projects", project, "log/time.txt")
    
    with open(fname, "r") as f:
        lines = f.readlines()
        
    lines = [l for l in lines if l.split()[0] != 'tmp']
    
    t_dict = {}
    s_dict = {}
    for l in lines:
        s_dict[l.split()[0]] = 0
        t_dict[l.split()[0]] = 0
        
    for l in lines:
        s_dict[l.split()[0]] += 1
        t_dict[l.split()[0]] += float(l.split()[1])
    
        
    t_list = []
    labels = []
    total = 0
    for key in s_dict:
        t_dict[key] = t_dict[key] / s_dict[key]
        t_list.append(t_dict[key])
        total += t_dict[key]
        labels.append(key)
            
    # Data to plot
    colors = range(len(t_list))

    # Plot
    plt.figure(figsize=(6,3), dpi=180)
    
    plt.subplot(1,2,1)
    plt.pie(t_list, labels=labels, 
            autopct=lambda(p): str(np.round(p * total / 100, 2)) + ' sec')

    plt.axis('equal')
    
    with open(fname, "r") as f:
        lines = f.readlines()
    
    # time seq
    freq = 0
    keys = []
    for l in lines:
        key = l.split()[0]
        if key == 'tmp':
            continue
        key = l.split()[0]
        if key not in keys:
            freq += 1
        else:
            break
        keys.append(key)
        
    ts = [float(t[4:-1]) for t in lines[::freq*2]]
    t_abs = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
    t_clk = [datetime.datetime.fromtimestamp(t) for t in ts[:-1]]
    
    ax = plt.subplot(1,2,2)
    ax.plot(t_clk, t_abs, c='r', alpha = 0.2)
    mean_t_abs = np.convolve(t_abs, np.ones((50,))/50)
    ax.plot(t_clk, mean_t_abs[:len(t_clk)], c='r')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        
    plt.tight_layout()
    plt.show()
    
    
def gpu(max_last_mod=120, lj=25):
    
    print "name".ljust(30), "host".ljust(20), "epoch".ljust(15), "gpu"
    print "------------------------------------------------------------------------------"
    
    n_projects_running = 0
    for p in sorted(os.listdir("projects")):
        fname = os.path.join("projects", p, "log/logs.txt")
        t = time.time() - os.path.getmtime(fname)
        if  t < max_last_mod:
            
            # process name
            print p.ljust(30),
            
            # host name
            fname = os.path.join("projects", p, "log/host_name.txt")
            if os.path.isfile(fname):
                with open(fname, 'r') as f:
                    host_name = f.read()
            else:
                host_name = "-"
            print host_name.ljust(20),
            
            # epoch
            fname = "projects/" + p + "/log/iters_per_epoch"
            iters_per_epoch = float(np.loadtxt(fname))
            fname = os.path.join("projects/", p, 'log/logs.txt')
            n_iters = len(np.loadtxt(fname, skiprows=1, delimiter=" "))
            epoch = n_iters / float(iters_per_epoch)
            print str(np.round(epoch, 2)).ljust(15),
            
            # gpu
            try:
                fname = "projects/" + p + "/log/gpu_id.txt"
                gpu_idx = int(np.loadtxt(fname))
                print gpu_idx,
            except:
                pass
            
            print " "
            n_projects_running += 1
    print "------------------------------------------------------------------------------"
    print "\n"
    print "{} projects running".format(n_projects_running)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def tiplet_nn(imgs_real, 
              y, 
              project, 
              epoch,
              sampler = None,
              n_pts_tsne = 4000,
              n_neighbors = 5,
              alpha = 0.3,
              real_img_mode = "single"):
    
    import matplotlib.pylab as plt
    
    # load models
    enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
    gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")
    dis = ld_gan.utils.model_handler.load_model(project, epoch, "dis")
    
    epoch_str = str(epoch).zfill(4)
    fname = "projects/" + project + "/tsne_pts/" \
                      + epoch_str + "_" + str(n_pts_tsne) + ".npy"

    try:
        z_mapped = np.load(fname)

    except:
        print "compute tsne..."
        f_X = ld_gan.utils.model_handler.apply_model(enc, 
                                                     imgs_real[:n_pts_tsne],
                                                     1000)
        tsne = TSNE(n_components=2, 
                    random_state=0, 
                    metric = 'cosine', 
                    learning_rate=1000)
        z_mapped = tsne.fit_transform(f_X)
        try:
            os.mkdir("projects/" + project + "/tsne_pts")
        except:
            pass
        np.save(fname, z_mapped)

    nbrs = NearestNeighbors(n_neighbors = 6).fit(z_mapped)
    
    
    # load fig
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize = (10.,10.))
    gs = gridspec.GridSpec(8, 11)
    ax1 = fig.add_subplot(gs[0:2, 0:3])
    ax2 = fig.add_subplot(gs[0:2, 3:6])
    ax3 = fig.add_subplot(gs[0:2, 6:9])
    ax2.axis('off')
    ax3.axis('off')
    axs0, axs1, axs2, axs3, axs4 = [], [], [], [], []
    for i in range(7):
        axs0.append(fig.add_subplot(gs[3, i]))
        axs1.append(fig.add_subplot(gs[4, i]))
        axs2.append(fig.add_subplot(gs[5, i]))
        axs3.append(fig.add_subplot(gs[6, i]))
        axs4.append(fig.add_subplot(gs[7, i]))
    paxs0 = fig.add_subplot(gs[3, 8:])
    paxs1 = fig.add_subplot(gs[4, 8:])
    paxs2 = fig.add_subplot(gs[5, 8:])
    paxs3 = fig.add_subplot(gs[6, 8:])
    paxs4 = fig.add_subplot(gs[7, 8:])
    axs = [axs0, axs1, axs2, axs3, axs4]
    paxs = [paxs0, paxs1, paxs2, paxs3, paxs4]
    
    for a1 in axs:
        for a2 in a1:
            a2.axis('off')

    ax1.scatter(z_mapped[:, 0], z_mapped[:, 1], c = y[:len(z_mapped)], 
                s = 10, alpha = alpha, edgecolors='none')
    
    
    def onclick(event):
        
        dists, idxs = nbrs.kneighbors(np.array([[event.xdata, event.ydata]]))
        idx = idxs[0][0]

        imgs = imgs_real[idxs[0]]

        for idx, img in enumerate(imgs[1:]):
            axs[idx][0].imshow(imgs[0])
            axs[idx][-1].imshow(img)
            ipt = np.array([imgs[0], img])
            z0, z1 = ld_gan.utils.model_handler.apply_model(enc, ipt)
            n_interpol = 5
            fac0 = [1 - (i+1)/float(n_interpol+1) for i in range(n_interpol)]
            fac1 = [(i+1)/float(n_interpol+1) for i in range(n_interpol)]
            zs = [z0*f0 + z1*f1 for f0, f1 in zip(fac0, fac1)]
            zs = np.array(zs)
            interpol_imgs = ld_gan.utils.model_handler.apply_model(gen, zs)
            dss = []
            for ii_idx, ii in enumerate(interpol_imgs):
                axs[idx][ii_idx+1].imshow(ii)
                ds = ld_gan.utils.model_handler.apply_model(dis, np.array([ii]))
                axs[idx][ii_idx+1].set_title(np.round(ds,3), fontsize=10)
                dss.append(float(ds))
            paxs[idx].clear()
            paxs[idx].plot(range(len(dss)), dss)
            paxs[idx].set_ylim([0, 0.5])
            for tick in paxs[idx].xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
            for tick in paxs[idx].yaxis.get_major_ticks():
                tick.label.set_fontsize(6)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
    
    
def pre_compute_tsne(project, imgs_real, n_pts_tsne=4000, epochs=None):
    
    epochs = epochs if epochs is not None else np.arange(0,1000,50)
    
    for epoch in tqdm(epochs):
        
        enc = ld_gan.utils.model_handler.load_model(project, epoch, "enc")
        gen = ld_gan.utils.model_handler.load_model(project, epoch, "gen")

        f_X = ld_gan.utils.model_handler.apply_model(enc, imgs_real, 1000)


        epoch_str = str(epoch).zfill(4)
        fname = "projects/" + project + "/tsne_pts/" \
                          + epoch_str + "_" + str(n_pts_tsne) + ".npy"

        try:
            z_mapped = np.load(fname)

        except:
            tsne = TSNE(n_components=2, 
                        random_state=0, 
                        metric = 'cosine', 
                        learning_rate=1000)
            z_mapped = tsne.fit_transform(f_X[:n_pts_tsne])
            try:
                os.mkdir("projects/" + project + "/tsne_pts")
            except:
                pass
            np.save(fname, z_mapped)
    
    
    
    
    
    
    