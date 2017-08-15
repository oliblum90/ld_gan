import os
import sys
import shutil

def init_project(name, ask_before_del = True):
    
    try:
        os.mkdir(name)
    except:
        
        if ask_before_del:
            decission = input('Delete project "{}"? (1 = yes, 2 = no)'.format(name))
        else:
            decission = 1
            
        if decission == 1:
            shutil.rmtree(name)
            os.mkdir(name)
        else:
            print "end programm"
            sys.exit()

            
def save_setup(name):
        
    import __main__ as main
    from shutil import copyfile
        
    copyfile(main.__file__, 
             os.path.join(name, main.__file__))