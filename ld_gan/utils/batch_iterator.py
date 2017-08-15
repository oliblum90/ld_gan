from sklearn.utils import shuffle
                
def get_data_iterator(batch_size, *args):
    
    n_samples = len(args[0])
    n_iters   = n_samples / batch_size
    
    idx_beg = 0
    idx_end = batch_size

    while True:
        
        idx_beg = idx_beg + batch_size
        idx_end = idx_end + batch_size
        
        if idx_end <= n_samples:
            
            yield tuple([arg[idx_beg:idx_end] for arg in args])
            
        else:
            
            idx_beg = 0
            idx_end = batch_size
            if len(args) == 1:
                args = (shuffle(*args), )
            else:
                args = shuffle(*args)
            
            
