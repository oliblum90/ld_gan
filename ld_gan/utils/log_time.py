import os
from time import time
import __main__ as main


def log_time(name):
    
    try:
        t = time()

        log_fname = os.path.join("projects", 
                                 main.__file__, 
                                 "log", 
                                 "time.txt")

        try:
            with open(log_fname, "r") as f:
                lines = f.readlines()

            if lines[-1].split(' ')[0] == "tmp":
                t = time() - float(lines[-1].split(' ')[1])
                line = "\n" + name + " " + str(t)
            else:
                line = "\n" + "tmp" + " " + str(t)

        except:
            line = "tmp" + " " + str(t)

        with open(log_fname, "a") as f:
            f.write(line)
    except:
        #print "ERROR in log_time"
        pass

    