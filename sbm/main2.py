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

no = 10
res = np.zeros((no,), dtype=float)

for rr in np.linspace(1, 3, 200):
    for hue in range(no):
        r = 2 
        m = 400
        A, labels = func.create_sbm(n=1000,cin=20,cout=1,q=r)
        #A, labels = W, labels_true
        constraint = func.create_constraint(labels, m)
        labels1 = func.unnorm_spec_clustering(A,r)
        labels2 = func.bethe_hessian_clustering(A,r)
        idx = func.fastge3 (A, r, constraint, r = rr)
        #idx = func.fastge2 (A, r, constraint)
        #print(func.nmi(labels,labels1))
        #print(func.nmi(labels,labels2))
        res[hue] = func.nmi(labels, idx)
        #print(func.nmi(labels, idx), end=', ')
    print(res.mean(), end=", ")
    print(rr)
