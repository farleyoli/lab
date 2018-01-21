import functions
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

c = 3 # parameter for SBM
size = 1000 # size of graph
no = 5 # number of times clustering is performed (and averaged) 
no_alpha = 200 # number of points for x-axis, depends on how long it would take

q_range = [2, 3, 5]
theoretical_values = [q_range[0]*math.sqrt(3), q_range[1]*math.sqrt(3), q_range[2]*math.sqrt(3)]

percentage = [0.25, 0.50, 0.75]

# 1: unnormalized, 2: bethe, 3: modfied both, 4: modified G, 5: modified H (eu sei, ta feio, mas foda-se)
res1 = np.zeros((len(percentage), no), dtype=float)
res2 = np.zeros((len(percentage), no), dtype=float)
#res3 = np.zeros((len(percentage), len(q_range), no), dtype=float)
#res4 = np.zeros((len(percentage), len(q_range), no), dtype=float)
res5 = np.zeros((len(percentage), len(q_range), no), dtype=float)

result1 = np.zeros((len(percentage), no_alpha), dtype=float)
result2 = np.zeros((len(percentage), no_alpha), dtype=float)
#result3 = np.zeros((len(percentage), len(q_range), no_alpha), dtype=float)
#result4 = np.zeros((len(percentage), len(q_range), no_alpha), dtype=float)
result5 = np.zeros((len(percentage), len(q_range), no_alpha), dtype=float)

for k in range(len(q_range)):
    alpha_range = np.linspace( 0.75 * theoretical_values[k], 1.5 * q_range[k] * theoretical_values[k], no_alpha)
    for alpha_index in range(len(alpha_range)):
        alpha = alpha_range[alpha_index]
        for hue in range(no):
            # compute cin and cout
            cin = (alpha + 3)/2
            cout = (3 - alpha)/2

            # create SBM graph
            A, labels = functions.create_sbm(n = size, cin = cin, cout = cout, q = q_range[k])
            

            labels1 = functions.unnorm_spec_clustering(A, q_range[k])
            labels2 = functions.bethe_hessian_clustering(A, q_range[k], r = np.sqrt(3))

            res1[k, hue] = functions.nmi(labels, labels1)
            res2[k, hue] = functions.nmi(labels, labels2)

            for p in range(len(percentage)):
                # create constraints 
                no_constraints = int(size * (percentage[p]))
                constraint = functions.create_constraint(labels, no_constraints)
                #labels3 = functions.fastge3 (A, q_range[k], constraint, r = np.sqrt(3), choice = 3)
                #labels4 = functions.fastge3 (A, q_range[k], constraint, r = np.sqrt(3), choice = 4)
                labels5 = functions.fastge3 (A, q_range[k], constraint, r = np.sqrt(3), choice = 5)
                #res3[p, k, hue] = functions.nmi(labels, labels3)
                #res4[p, k, hue] = functions.nmi(labels, labels4)
                res5[p, k, hue] = functions.nmi(labels, labels5)
            
            print("hue", end=": "); print(hue, end=", ")
            print("k", end=": "); print(q_range[k], end=", ")
            print("alpha", end=": "); print(alpha)

        # 1: unnormalized, 2: bethe, 3: modfied both, 4: modified G, 5: modified H (eu sei, ta feio, mas foda-se)
        result1[k, alpha_index] = res1[k,:].mean()
        print("unnormalized", end=": "); print(result1[k, alpha_index])
        result2[k, alpha_index] = res2[k, :].mean()
        print("bethe", end=": "); print(result2[k, alpha_index])
        for p in range(len(percentage)):
            #result3[p, k, alpha_index] = res3[p,k,:].mean()
            #print(100*percentage[p], end="%, "); print("fastge2 (both)", end=": "); print(result3[p, k, alpha_index])
            #result4[p, k, alpha_index] = res4[p,k,:].mean()
            #print(100*percentage[p], end="%, ");print("fastge2 (N)", end=": "); print(result4[p, k, alpha_index])
            result5[p, k, alpha_index] = res5[p,k,:].mean()
            print(100*percentage[p], end="%, ");print("fastge2 (H)", end=": "); print(result5[p, k, alpha_index])
        print("")
