import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.io import loadmat

# get data
x = loadmat('matlab.mat')
x1 = x['x'][3,0].ravel()
x2 = x['x'][7,0].ravel()
y1 = x['y'][3,0].ravel()
y2 = x['y'][7,0].ravel()

# regression
regr1 = sm.OLS(y1, sm.add_constant(x1))
res1 = regr1.fit()
st1, data1, ss21 = summary_table(res1, alpha=0.05)
regr2 = sm.OLS(y2, sm.add_constant(x2))
res2 = regr2.fit()
st2, data2, ss22 = summary_table(res2, alpha=0.05)

# Get the confidence intervals of the model (95% bound!!!)
predict_mean_ci_low1, predict_mean_ci_upp1 = data1[:,4:6].T
predict_mean_ci_low2, predict_mean_ci_upp2 = data2[:,4:6].T

# plot graph
plt.plot(x1, y1, label='CVQE', color = "blue")
plt.plot(x1, data1[:,2], color = "blue")
plt.fill_between(x1, predict_mean_ci_low1, predict_mean_ci_upp1, color = 'blue', alpha = 0.4, label = 'Confidence Bounds of CVQE')
plt.plot(x2, y2, label='FAST-GE-2.0', color = "orange")
plt.plot(x2, data2[:,2], color = "orange")
plt.fill_between(x2, predict_mean_ci_low2, predict_mean_ci_upp2, color = 'orange', alpha = 0.4, label = 'Confidence Bounds of FAST-GE-2.0')
plt.legend(loc='best')
plt.title('Constrained Clustering using CVQE and FAST-GE-2.0')
plt.xlabel(r'$n_c$: Number of constrains used')
plt.ylabel(r'Normalized Mutual Information')
plt.show()
