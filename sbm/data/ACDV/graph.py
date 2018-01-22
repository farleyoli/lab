import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from statsmodels.stats.outliers_influence import summary_table
from scipy.io import loadmat

no_alpha = 200 # number of points for x-axis, depends on how long it would take
q_range = [2, 3, 5]
theoretical_values = [q_range[0]*math.sqrt(3), q_range[1]*math.sqrt(3), q_range[2]*math.sqrt(3)]
q = 2 

# fetch results
result = np.load('result.npz')
y1 = result['arr_0'][q]
y2 = result['arr_1'][q]
y3 = result['arr_2'][0,q]
y4 = result['arr_2'][1,q]
y5 = result['arr_2'][2,q]
x =  alpha_range = np.linspace( 0.75 * theoretical_values[q], 1.5 * q_range[q] * theoretical_values[q], no_alpha)

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

plt.plot(x, y3, label=r'Modified FAST-GE-2.0: 25%', color = "green")
#plt.plot(x, data3[:,2], color = "green")
#plt.fill_between(x, predict_mean_ci_low3, predict_mean_ci_upp3, color = 'green', alpha = 0.4) 

plt.plot(x, y4, label='Modified FAST-GE-2.0: 50%', color = "red")
#plt.plot(x, data4[:,2], color = "red")
#plt.fill_between(x, predict_mean_ci_low4, predict_mean_ci_upp4, color = 'red', alpha = 0.4) 

plt.plot(x, y5, label='Modified FAST-GE-2.0: 75%', color = "purple")
#plt.plot(x, data5[:,2], color = "purple")
#plt.fill_between(x, predict_mean_ci_low5, predict_mean_ci_upp5, color = 'purple', alpha = 0.4) 

plt.axvline(x = theoretical_values[q], linewidth = 2, color="black")
#plt.text(theoretical_values[q]-3, -1, 'Theoretical Value', fontsize=12)
plt.annotate('Theoretical Value', (0,0), (25, -20), xycoords='axes fraction', textcoords='offset points', va='top')

plt.legend(loc='best')
plt.title(r'NMI of different clustering methods as a function of $c_{in}-c_{out}$ for 5 clusters')
plt.xlabel(r'$c_{in}-c_{out}$')
plt.ylabel(r'Normalized Mutual Information')
plt.show()


