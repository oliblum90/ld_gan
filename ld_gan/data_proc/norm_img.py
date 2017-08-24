
MODE = 0

def norm(imgs, mode = MODE):
    if mode == 0:
        return (imgs.astype('float32') / 127.5) - 1.
    if mode == 1:
        return imgs.astype('float32') / 255.
    
    
def un_norm(imgs, mode = MODE):
    if mode == 0:
        return ((imgs + 1.) * 127.5).astype('uint8')
    if mode == 1:
        imgs[imgs < 0] = 0
        return (imgs * 255.).astype('uint8')