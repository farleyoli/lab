import matplotlib.pyplot as plt
from scipy.io import loadmat
%matplotlib
x = loadmat('matlab.mat')
x1 = x['x'][3,0].ravel()
x2 = x['x'][7,0].ravel()
y2 = x['y'][7,0].ravel()
y1 = x['x'][3,0].ravel()
y1 = x['y'][3,0].ravel()
plt.hold(True)
plt.plot(x1, y1)
plt.hold(True)
plt.plot(x2, y2)
%save test.py
%save test.py _
%save test.py _oh
import readline
readline.write_history_file('test.py')
readline.write_history_file('test.py')
%history -f /tmp/history.py
%history -f history.py
