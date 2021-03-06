
MODE = 0

def norm(imgs, mode = MODE):
    if mode == 0:
        return (imgs.astype('float32') / 127.5) - 1.
    if mode == 1:
        return imgs.astype('float32') / 255.
    if mode == 2:
        return norm_mean_std(imgs)
    
    
def un_norm(imgs, mode = MODE):
    if mode == 0:
        return ((imgs + 1.) * 127.5).astype('uint8')
    if mode == 1:
        imgs[imgs < 0] = 0
        return (imgs * 255.).astype('uint8')
    if mode == 2:
        return un_norm_mean_std(imgs)
    
    
def norm_mean_std(imgs, 
                  mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225]):
    
    imgs = imgs.astype('float32')
    imgs = imgs / 255.
    
    for c in range(3):
        imgs[:,:,c] = (imgs[:,:,c] - mean[c]) / std[c]
        
    return imgs


def un_norm_mean_std(imgs, 
                     mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225]):
    
    for c in range(3):
        imgs[:,:,c] = imgs[:,:,c] * std[c] + mean[c]
        
    imgs = imgs * 255.
    imgs = img.astype('uint8')
        
    return imgs

