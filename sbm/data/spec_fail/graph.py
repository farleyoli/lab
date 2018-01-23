import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.io import loadmat

# fetch results
result = np.load('spec_fail.npz')
y1 = result['arr_0'][result['arr_0'] != 0].copy()
y2 = result['arr_1'][result['arr_1'] != 0].copy()
n = len(y1)
#x = np.arange(51, n+51, 1)
x = np.linspace(7,50,n)

# regression
regr1 = sm.OLS(y1, sm.add_constant(x))
res1 = regr1.fit()
st1, data1, ss1 = summary_table(res1, alpha=0.05)
regr2 = sm.OLS(y2, sm.add_constant(x))
res2 = regr2.fit()
st2, data2, ss2 = summary_table(res2, alpha=0.05)


# Get the confidence intervals of the model (95% bound!!!)
predict_mean_ci_low1, predict_mean_ci_upp1 = data1[:,4:6].T
predict_mean_ci_low2, predict_mean_ci_upp2 = data2[:,4:6].T

# plot graphs
plt.plot(x, y1, label='Unnormalized Spectral Clustering', color = "blue")
#plt.plot(x, data1[:,2], color = "blue")
#plt.fill_between(x, predict_mean_ci_low1, predict_mean_ci_upp1, color = 'blue', alpha = 0.4)

#plt.plot(x[0:130], y2[0:130], label='Constrained Clustering with Bethe Hessian', color = "orange")
#plt.plot(x, data2[:,2], color = "orange")
#plt.fill_between(x, predict_mean_ci_low2, predict_mean_ci_upp2, color = 'orange', alpha = 0.4)

#plt.plot(x, y3, label=r'Modified FAST-GE-2.0: Both $L_N$ and $L_H$', color = "green")
#plt.plot(x, data3[:,2], color = "green")
#plt.fill_between(x, predict_mean_ci_low3, predict_mean_ci_upp3, color = 'green', alpha = 0.4) 

#plt.plot(x, y4, label='Modified FAST-GE-2.0: Only $L_N$', color = "red")
#plt.plot(x, data4[:,2], color = "red")
#plt.fill_between(x, predict_mean_ci_low4, predict_mean_ci_upp4, color = 'red', alpha = 0.4) 

#plt.plot(x, y5, label='Modified FAST-GE-2.0: Only $L_H$', color = "purple")
#plt.plot(x, data5[:,2], color = "purple")
#plt.fill_between(x, predict_mean_ci_low5, predict_mean_ci_upp5, color = 'purple', alpha = 0.4) 

plt.legend(loc='best')
plt.title('Failure of Unnormalized Spectral Clustering')
plt.xlabel(r'$c_{in}$')
plt.ylabel(r'Normalized Mutual Information')
plt.show()


