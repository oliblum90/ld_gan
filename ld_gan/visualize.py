import os
import inspect, re
import numpy as np
# import cv2
import scipy.misc
from matplotlib import gridspec
import matplotlib
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from data_proc.transformer import np_to_tensor, tensor_to_np
import ld_gan



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
    
            
def disp(img_list, title_list = None, fname = None):
    """
    display a list of images
    """
    import matplotlib.pylab as plt

    plt.figure()

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
                      high_res        = False):
    
    import matplotlib.pylab as plt
    matplotlib.rcParams.update({'font.size': 8})
    
    try:
        fname = "projects/" + project + "/log/iters_per_epoch"
        iters_per_epoch = float(np.loadtxt(fname))
    except:
        print "no iters_per_epoch file"
    
    fname = os.path.join(path, project, 'log/logs.txt')
    logs = np.loadtxt(fname, skiprows=1, delimiter=" ")
    labels = np.genfromtxt('projects/x_test.py/log/logs.txt',dtype='str')[0]
    colors = ['g', 'b', 'r', 'gold', 'k', 'magenta'][:len(labels)]

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
        ax1.plot(x, logs[:,idx], c=colors[idx], alpha = 0.2)
        y = np.convolve(logs[:,idx], 
                        np.ones((smooth,))/smooth, 
                        mode='valid')
        l1 = ax1.plot(xc, y, c=colors[idx], label=labels[idx])
        ymax_temp = max(ymax_temp, y.max())

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
    gen_tsne_epoch = int(os.listdir(path_ht)[-1][:4])
    
    # show fake image    
    fname = str(gen_img_epoch).zfill(4) + "_fake.png"
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
        
        imgs = os.listdir(path_ht)
        iters = [int(img_str[:4]) for img_str in imgs]
        idx = np.argmin(np.abs(np.array(iters) - event.xdata))
        gen_tsne_epoch = iters[idx]
        
        fname = str(gen_img_epoch).zfill(4) + "_fake.png"
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
                       n_clusters = 30):
    
    if fname is not None:
        matplotlib.use('Agg')
        import matplotlib.pylab as plt
    else:
        import matplotlib.pylab as plt
    
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
    if y is None:
        ax2.scatter(X_tsne[:,0], X_tsne[:,1])
    else:
        ax2.scatter(X_tsne[:,0], X_tsne[:,1], c=y[:n_pts_tsne])
        
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
                       gen_epoch,
                       z_mapped = None, 
                       sampler = None,
                       n_pts_tsne = 4000,
                       n_neighbors = 5,
                       alpha = 0.003):
    
    import matplotlib.pylab as plt
    from keras.models import load_model
    import tensorflow as tf
    
    # load models
    epoch_str = str(gen_epoch).zfill(4)
    gen = torch.load("projects/" + project + "/model/g_" + epoch_str + ".pth")
    try:
        enc = torch.load("projects/" + project + "/model/e_" + epoch_str + ".pth")
    except:
        enc = torch.load("projects/" + project + "/model/enc.pth")
    
    if z_mapped is None:
        
        fname = "projects/" + project + "/tsne_pts/" \
                          + epoch_str + "_" + str(n_pts_tsne) + ".npy"
        
        try:
            z_mapped = np.load(fname)
        
        except:
            print "compute tsne..."
            f_X = enc(imgs_real[:n_pts_tsne])
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

    def onclick(event):
        dists, idxs = nbrs.kneighbors(np.array([event.xdata, event.ydata]))
        idx = idxs[0][0]
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, idx, event.y, event.xdata, event.ydata))

        ax2.set_title("real img")
        img_real = np.squeeze(imgs_real[idx])
        ax2.imshow(img_real, cmap='gray')
        
        z_encs = []
        for idx in idxs[0]:
            z_enc = enc(np.array([imgs_real[idx]]))
            z_encs.append(z_enc)
        z_enc = np.mean(np.array(z_encs), axis = 0)
        
        # idx = idxs[0][0]
        # z_enc = enc.predict(np.array([imgs_real[idx]]))
        
        if sampler is not None:
            z_enc = sampler(z_enc)
        img_fake = gen(z_enc)
        
        ax3.set_title("fake img")
        img_fake = np.squeeze(img_fake)
        ax3.imshow(img_fake, cmap='gray')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
def move_in_z_space(project, gen_epoch, img1, img2, enc_epoch = None, sampler = None):
    
    import matplotlib.pylab as plt
    from keras.models import load_model
    import tensorflow as tf
    from matplotlib.widgets import Slider, Button, RadioButtons
    
    # load models
    epoch_str = str(gen_epoch).zfill(4)
    enc_epoch_str = str(enc_epoch).zfill(4)
    gen = load_model("projects/" + project + "/model/gen.h5")
    gen.load_weights("projects/" + project + "/model/g_" + epoch_str + ".h5")
    enc = load_model("projects/" + project + "/model/enc.h5", custom_objects={"tf": tf})
    if enc_epoch is not None:
        enc.load_weights("projects/" + project + "/model/e_" + enc_epoch_str + ".h5")
    else:
        enc.load_weights("projects/" + project + "/model/enc_w_0.h5")
    
    # compute z-start and z-end
    z0, z1 = enc.predict(np.array([img1, img2]))
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

    img = gen.predict(np.array([z0]))
    img = np.squeeze(img[idx])
    pic = ax.imshow(img, cmap = 'gray')
    ax.axis('off')

    slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axis_color)
    slider = Slider(slider_ax, 'z', 0.0, 1.0, valinit=0.0)

    def sliders_on_changed(val):
        z = z0 + (z1 - z0) * slider.val
        img = gen.predict(np.array([z]))
        img = np.squeeze(img)
        pic.set_data(img)
        # pic.set_data(X[int(slider.val),:,:,0])
        fig.canvas.draw_idle()
    slider.on_changed(sliders_on_changed)

    plt.show()
    


def img_to_img(imgs, project, gen_epoch, enc_epoch = None):
    
    from keras.models import load_model
    import tensorflow as tf
    from matplotlib.widgets import Slider, Button, RadioButtons
    
    # load models
    epoch_str = str(gen_epoch).zfill(4)
    enc_epoch_str = str(enc_epoch).zfill(4)
    gen = load_model("projects/" + project + "/model/gen.h5")
    gen.load_weights("projects/" + project + "/model/g_" + epoch_str + ".h5")
    enc = load_model("projects/" + project + "/model/enc.h5", custom_objects={"tf": tf})
    if enc_epoch is not None:
        enc.load_weights("projects/" + project + "/model/e_" + enc_epoch_str + ".h5")
    else:
        enc.load_weights("projects/" + project + "/model/enc_w_0.h5")
        
    fake_imgs = gen.predict(enc.predict(imgs))
            
    return fake_imgs
    
    
    
    
