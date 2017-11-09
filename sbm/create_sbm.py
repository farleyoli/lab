import numpy as np
def create_sbm (n, cin, cout):
    pin = cin/n 
    pout = cout/n
    s = (n,n)
    A = np.zeros(s)
