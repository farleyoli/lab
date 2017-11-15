import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy.cluster.vq import kmeans2,vq

def create_sbm (n = 1000, cin = 7, cout = 1):
    pin = float(cin)/float(n) 
    pout = float(cout)/float(n)
    labels = np.random.randint(2, size=n)
    A = np.zeros((n,n), dtype=int)
    
    for (i,j), value in np.ndenumerate(A):
        if i >= j:
            if labels[i] != labels[j]:
                if np.random.rand() <= pout:
                    A[i,j] = 1
                    A[j,i] = 1
            else:
                if np.random.rand() <= pin:
                    A[i,j] = 1
                    A[j,i] = 1
    return A, labels

def unnorm_spec_clustering (W, k):
    d = []
    for i in range(W.shape[1]):
        d.append(sum(W[i,:]))
    D = np.diag(d)
    L = D - W 
    E,V = LA.eigh(a=L,eigvals=(0,k-1))
    centroids, labels = kmeans2(V,k)
    return labels 
