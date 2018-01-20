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
# 1: unnormalized, 2: bethe, 3: modfied both, 4: modified G, 5: modified H
res1 = np.zeros((no,), dtype=float)
res2 = np.zeros((no,), dtype=float)
res3 = np.zeros((no,), dtype=float)
res4 = np.zeros((no,), dtype=float)
res5 = np.zeros((no,), dtype=float)
result1 = np.zeros((size,), dtype=float)
result2 = np.zeros((size,), dtype=float)
result3 = np.zeros((size,), dtype=float)
result4 = np.zeros((size,), dtype=float)
result5 = np.zeros((size,), dtype=float)
for m in np.arange(200, size-10, 10):
    print('m: ', end=''); print(m)
    for hue in range(no):
        k = 2 
        A, labels = functions.create_sbm(n = size, cin = 7, cout = 1, q = k)
        #A, labels = W, labels_true
        constraint = functions.create_constraint(labels, m)
        labels1 = functions.unnorm_spec_clustering(A, k)
        labels2 = functions.bethe_hessian_clustering(A, k, r = np.sqrt(3))
        labels3 = functions.fastge3 (A, k, constraint, r = np.sqrt(3), choice = 3)
        labels4 = functions.fastge3 (A, k, constraint, r = np.sqrt(3), choice = 4)
        labels5 = functions.fastge3 (A, k, constraint, r = np.sqrt(3), choice = 5)
        res1[hue] = functions.nmi(labels, labels1)
        res2[hue] = functions.nmi(labels, labels2)
        res3[hue] = functions.nmi(labels, labels3)
        res4[hue] = functions.nmi(labels, labels4)
        res5[hue] = functions.nmi(labels, labels5)
        #print(functions.nmi(labels, idx), end=', ')
        #print('hue:', end=' ')
        print('hue: ', end=''); print(hue)
    result1[m] = res1.mean()
    print(result1[m])
    result2[m] = res2.mean()
    print(result2[m])
    result3[m] = res3.mean()
    print(result3[m])
    result4[m] = res4.mean()
    print(result4[m])
    result5[m] = res5.mean()
    print(result5[m])
