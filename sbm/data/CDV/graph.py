import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.io import loadmat

# fetch results
result = np.load('CDV_all.npz')
y1 = result['arr_0'][result['arr_0'] != 0].copy()
y1[:] = y1.mean()
y2 = result['arr_1'][result['arr_1'] != 0].copy()
y2[:] = y2.mean()
y3 = result['arr_2'][result['arr_2'] != 0].copy()
y4 = result['arr_3'][result['arr_3'] != 0].copy()
y5 = result['arr_4'][result['arr_4'] != 0].copy()
x = np.arange(200, 990, 10)

# regression
regr1 = sm.OLS(y1, sm.add_constant(x))
res1 = regr1.fit()
st1, data1, ss1 = summary_table(res1, alpha=0.05)
regr2 = sm.OLS(y2, sm.add_constant(x))
res2 = regr2.fit()
st2, data2, ss2 = summary_table(res2, alpha=0.05)
regr3 = sm.OLS(y3, sm.add_constant(x))
res3 = regr3.fit()
st3, data3, ss3 = summary_table(res3, alpha=0.05)
regr4 = sm.OLS(y4, sm.add_constant(x))
res4 = regr4.fit()
st4, data4, ss4 = summary_table(res4, alpha=0.05)
regr5 = sm.OLS(y5, sm.add_constant(x))
res5 = regr5.fit()
st5, data5, ss5 = summary_table(res5, alpha=0.05)


# Get the confidence intervals of the model (95% bound!!!)
predict_mean_ci_low1, predict_mean_ci_upp1 = data1[:,4:6].T
predict_mean_ci_low2, predict_mean_ci_upp2 = data2[:,4:6].T
predict_mean_ci_low3, predict_mean_ci_upp3 = data3[:,4:6].T
predict_mean_ci_low4, predict_mean_ci_upp4 = data4[:,4:6].T
predict_mean_ci_low5, predict_mean_ci_upp5 = data5[:,4:6].T

# plot graphs
plt.plot(x, y1, label='Unnormalized Spectral Clustering', color = "blue")
#plt.plot(x, data1[:,2], color = "blue")
#plt.fill_between(x, predict_mean_ci_low1, predict_mean_ci_upp1, color = 'blue', alpha = 0.4)

plt.plot(x, y2, label='Constrained Clustering with Bethe Hessian', color = "orange")
#plt.plot(x, data2[:,2], color = "orange")
#plt.fill_between(x, predict_mean_ci_low2, predict_mean_ci_upp2, color = 'orange', alpha = 0.4)

plt.plot(x, y3, label=r'Modified FAST-GE-2.0: Both $L_N$ and $L_H$', color = "green")
plt.plot(x, data3[:,2], color = "green")
plt.fill_between(x, predict_mean_ci_low3, predict_mean_ci_upp3, color = 'green', alpha = 0.4) 

plt.plot(x, y4, label='Modified FAST-GE-2.0: Only $L_N$', color = "red")
plt.plot(x, data4[:,2], color = "red")
plt.fill_between(x, predict_mean_ci_low4, predict_mean_ci_upp4, color = 'red', alpha = 0.4) 

plt.plot(x, y5, label='Modified FAST-GE-2.0 ', color = "purple")
plt.plot(x, data5[:,2], color = "purple")
plt.fill_between(x, predict_mean_ci_low5, predict_mean_ci_upp5, color = 'purple', alpha = 0.4) 

plt.legend(loc='best')
plt.title('Comparison of different modifications of FAST-GE-2.0')
plt.xlabel(r'$n_c$: Number of constrains used')
plt.ylabel(r'Normalized Mutual Information')
plt.show()


