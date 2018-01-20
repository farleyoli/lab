from __future__ import print_function
import functions
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
#reload(functions)
#centers = [[0, 0], [-1, -1]]
#n_clusters = len(centers)
#X, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.7)
#plt.plot(X)
#W = functions.compute_W (X)
#print(W)

size = 1000
no = 10 
res = np.zeros((no,), dtype=float)
result = np.zeros((size,), dtype=float)
for cin in np.arange(7, 50, 1):
    for n in range(no):
        k = 2 
        #m = 400
        A, labels = functions.create_sbm(n = size, cin = cin, cout = 1, q = k)
        #A, labels = W, labels_true
        #constraint = functions.create_constraint(labels, m)
        labels1 = functions.unnorm_spec_clustering(A, k)
        #labels2 = functions.bethe_hessian_clustering(A, k, r = np.sqrt(3))
        #idx = functions.fastge3 (A, r, constraint, r = rr)
        #idx = functions.fastge3 (A, k, constraint, r = np.sqrt(3))
        #idx = functions.fastge2 (A, r, constraint)
        #print(functions.nmi(labels,labels1))
        #print('bethe_hessian: ', end=''); print(functions.nmi(labels,labels2))
        res[n] = functions.nmi(labels, labels1)
        #print(functions.nmi(labels, idx), end=', ')
        #print('hue:', end=' ')
        #print(hue)
        print(n)
    result[cin] = np.mean(res)
    print(result[cin])
