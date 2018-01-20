import math
import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy.sparse import linalg as SLA
from scipy.cluster.vq import kmeans2
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

def create_sbm (n = 1000, cin = 7, cout = 1, q = 2):
    """ Uses the stochastic block model to generate a graph.
        Inputs:
        - n: number of nodes in the graph
        - cin, cout: parameters for the model
        - q: number of groups (bug? --> aparently not)
        Outputs:
        - A: adjacency matrix of the generated graph
        - labels: array containing the true label of each node
    """
    pin = float(cin)/float(n) 
    pout = float(cout)/float(n)
    labels = np.random.randint(q, size=n)
    A = np.zeros((n,n), dtype=int)
    
    for (i,j), value in np.ndenumerate(A):
        if i > j:
            if labels[i] != labels[j]:
                if np.random.rand() <= pout:
                    A[i,j] = 1
                    A[j,i] = 1
            else:
                if np.random.rand() <= pin:
                    A[i,j] = 1
                    A[j,i] = 1
    for i in range(n):
        A[i,i] = 1
    return A, labels

def unnorm_spec_clustering (W, k):
    """ Uses unnormalized spectral clustering to cluster nodes of a graph
        whose adjacency matrix is W into k classes.
        Outputs:
        - labels: array with the predicted label of each node
    """
    d = np.sum(W, axis=0)
    D = np.diag(d)
    L = D - W 
    E,V = LA.eigh(a=L,eigvals=(0,k-1))
    centroids, labels = kmeans2(V,k, minit='points')
    return labels 


def bethe_hessian_clustering (W, k, r = 0):
    """ Uses the Bethe hessian to cluster nodes of a graph whose
        adjacency matrix is W into k classes.
        Input:
        r: parameter for computing the bethe hessian
        Output:
        labels: array with the predicted label of each node
    """
    n = W.shape[0]
    d = np.sum(W, axis=0)
    D = np.diag(d)
    if r == 0:
        r = math.sqrt(3)
    H = ((r**2)-1)*np.eye(n) - r*W + D
    E,V = LA.eigh(a=H,eigvals=(0,k-1))
    centroids, labels = kmeans2(V,k, minit='points')
    return labels

def fastge2 (W, k, constraint):
    """ Format for constraints: constraint[i] = k iff
        the i-th datum is an element of Vj. In case it is not
        an element of any Vj, constraint[i] = -1.
        
        Outputs:
        - idx: 
    """
    n, m = W.shape
    mu = 0.001
    W = W.astype(float)
    Z = np.ones((n,1), dtype=float)
    Z = Z/math.sqrt(n)

    # Compute the Laplacian of the original graph
    d = np.sum(W, axis = 0)
    D = np.diag(d)
    L = D - W

    # Compute Wm
    Wm = np.zeros((n, m), dtype=float)
    for l in range(k):
        for i in range(n):
            for j in range(m):
                if ((constraint[i] == constraint[j]) and (constraint[i] == l) and (i != j)):
                    Wm[i,j] = ((d[i])*(d[j]))/((d.min())*(d.max()))

    # Compute Wh
    Wc = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if ((constraint[i] != constraint[j]) and (constraint[i] != -1) and (constraint[j] != -1)):
                Wc[i,j] = ((d[i])*(d[j]))/((d.min())*(d.max()))

    Kc = np.zeros((n, m), dtype=float)
    dc = np.sum(Wc, axis = 0)
    dcc = np.sum(Wc+Wc.T, axis = 0)
    for i in range(n):
        for j in range(m):
            Kc[i,j] = ((dcc[i])*(dcc[j]))/(np.sum(dcc))

    Wh = np.zeros((n, m), dtype=float)
    Wh = (Wc + Wc.T + Kc)/n

    # Compute Lg
    Wg = W + Wm
    dg = np.sum(Wg, axis=0)
    Dg = np.diag(dg)
    Lg = Dg - Wg

    # Compute Lh
    dh = np.sum(Wh, axis=0)
    Dh = np.diag(dh)
    Lh = Dh - Wh

    # Compute K
    K = (-1)*Lh

    # Compute M
    M = Lg + mu*Lh + np.matmul(Z, Z.T)


    # Solve general eigenvalue problem for pencil (K,M)
    Df, Vf = SLA.eigsh(K, k = k, M = M)


    # Renormalize Vf
    for i in range(k):
        Vf[:,i] = Vf[:,i]/LA.norm(Vf[:,i])
    for i in range(n):
        Vf[i,:] = Vf[i,:]/LA.norm(Vf[i,:])

    # k-means (last step)
    centroids, idx = kmeans2(Vf, k, minit='points')

    return idx

def create_constraint (old_constraint, num):
    n = old_constraint.shape[0]
    new_constraint = (-1)*np.ones((n,), dtype=int)
    while (num > 0):
        random = np.random.randint(n)
        if (new_constraint[random] != old_constraint[random]):
            new_constraint[random] = old_constraint[random]
            num -= 1
    return new_constraint

def compute_W (X, sigma = 2):
    n = X.shape[0]
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i,j] = np.exp(-np.power(LA.norm(X[i,:]-X[j,:]),2)/(2 * sigma * sigma))
    return W

def fastge3 (W, k, constraint, r = 3, choice = 5):
    """ Format for constraints: constraint[i] = k iff
        the i-th datum is an element of Vj. In case it is not
        an element of any Vj, constraint[i] = -1.
        
        Outputs:
        - idx: 
    """

    n, m = W.shape
    mu = 0.001
    W = W.astype(float)
    Z = np.ones((n,1), dtype=float)
    Z = Z/math.sqrt(n)

    # Compute the Laplacian of the original graph
    d = np.sum(W, axis = 0)
    D = np.diag(d)
    #L = D - W
    L = (r*r - 1)*np.eye(n) - r*W + D


    # Compute Wm
    Wm = np.zeros((n, m), dtype=float)
    for l in range(k):
        for i in range(n):
            for j in range(m):
                if ((constraint[i] == constraint[j]) and (constraint[i] == l) and (i != j)):
                    Wm[i,j] = ((d[i])*(d[j]))/((d.min())*(d.max()))

    # Compute Wh
    Wc = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if ((constraint[i] != constraint[j]) and (constraint[i] != -1) and (constraint[j] != -1)):
                Wc[i,j] = ((d[i])*(d[j]))/((d.min())*(d.max()))

    Kc = np.zeros((n, m), dtype=float)
    dc = np.sum(Wc, axis = 0)
    dcc = np.sum(Wc+Wc.T, axis = 0)
    for i in range(n):
        for j in range(m):
            Kc[i,j] = ((dcc[i])*(dcc[j]))/(np.sum(dcc))

    Wh = np.zeros((n, m), dtype=float)
    Wh = (Wc + Wc.T + Kc)/n

    # Compute Lg
    Wg = W + Wm
    dg = np.sum(Wg, axis=0)
    Dg = np.diag(dg)
    if choice == 5:
        Lg = Dg - Wg
    else:
        Lg = (r*r - 1)*np.eye(n) - r*Wg + Dg

    # Compute Lh
    dh = np.sum(Wh, axis=0)
    Dh = np.diag(dh)
    if choice == 4:
        Lh = Dh - Wh
    else:
        Lh = (r*r - 1)*np.eye(n) - r*Wh + Dh

    # Compute K
    K = (-1)*Lh

    # Compute M
    M = Lg + mu*Lh + np.matmul(Z, Z.T)


    # Solve general eigenvalue problem for pencil (K,M)
    Df, Vf = SLA.eigsh(K, k = k, M = M)


    # Renormalize Vf
    for i in range(k):
        Vf[:,i] = Vf[:,i]/LA.norm(Vf[:,i])
    for i in range(n):
        Vf[i,:] = Vf[i,:]/LA.norm(Vf[i,:])

    # k-means (last step)
    centroids, idx = kmeans2(Vf, k, minit='points')

    return idx
