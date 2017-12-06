from __future__ import print_function
import func
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
reload(func)
centers = [[0, 0], [-1, -1]]
n_clusters = len(centers)
#X, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.7)
#plt.plot(X)
#W = func.compute_W (X)
#print(W)

size = 1000
no = 10 
res = np.zeros((no,), dtype=float)
result = np.zeros((size,), dtype=float)
for m in np.arange(100, size, 10):
    print('m: ', end=''); print(m)
    for hue in range(no):
        k = 2 
        #m = 400
        A, labels = func.create_sbm(n = size, cin = 7, cout = 1, q = k)
        #A, labels = W, labels_true
        constraint = func.create_constraint(labels, m)
        #labels1 = func.unnorm_spec_clustering(A, k)
        #labels2 = func.bethe_hessian_clustering(A, k, r = np.sqrt(3))
        #idx = func.fastge3 (A, r, constraint, r = rr)
        idx = func.fastge3 (A, k, constraint, r = np.sqrt(3))
        #idx = func.fastge2 (A, r, constraint)
        #print(func.nmi(labels,labels1))
        #print('bethe_hessian: ', end=''); print(func.nmi(labels,labels2))
        res[hue] = func.nmi(labels, idx)
        #print(func.nmi(labels, idx), end=', ')
        #print('hue:', end=' ')
        #print(hue)
    result[m] = res.mean()
    print('fastge3: ', end=''); print(res.mean())
