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
hmm = np.linspace(7,40,300)


size = 1000
no = 10 
res = np.zeros((no,), dtype=float)
res2 = np.zeros((no,), dtype=float)
result = np.zeros((size,), dtype=float)
result2 = np.zeros((size,), dtype=float)
for cin in range(len(hmm)):
    for n in range(no):
        k = 2 
        #m = 400
        A, labels = functions.create_sbm(n = size, cin = hmm[cin], cout = 1, q = k)
        #A, labels = W, labels_true
        #constraint = functions.create_constraint(labels, m)
        labels1 = functions.unnorm_spec_clustering(A, k)
        labels2 = functions.bethe_hessian_clustering(A, k, r = np.sqrt(3))
        #idx = functions.fastge3 (A, r, constraint, r = rr)
        #idx = functions.fastge3 (A, k, constraint, r = np.sqrt(3))
        #idx = functions.fastge2 (A, r, constraint)
        #print(functions.nmi(labels,labels1))
        #print('bethe_hessian: ', end=''); print(functions.nmi(labels,labels2))
        res[n] = functions.nmi(labels, labels1)
        res2[n] = functions.nmi(labels, labels2)
        #print(functions.nmi(labels, idx), end=', ')
        #print('hue:', end=' ')
        #print(hue)
        print(n)
    result[cin] = np.mean(res)
    result2[cin] = np.mean(res2)
    print(result[cin])
    print(result2[cin])
