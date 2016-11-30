import matplotlib.pyplot as plt
import scipy.io 
import numpy as np
import obspy
from obspy.signal.detrend import polynomial
import os
os.chdir("C:\Work Related\Projects\Sunil\Hackathon")
ecg=scipy.io.loadmat('100m.mat')['val']
#ecg=np.reshape(ecg,len(ecg))
plt.plot(ecg)

ecgdt = polynomial(ecg[1], order=12)

import pywt

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(signal, order=(2, 1, 2))

from statsmodels.tsa.ar_model import AR

x = ar_mod.fit(5)
x.fittedvalues

from sklearn import svm

from sklearn import svm
X = [[0, 0], [1, 1]]
In[117]: y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
clf = svm.SVC(C=4000, coef0=5.6)
clf.fit(X, y) 
clf.predict([[0,0]])

#multiclass
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
#6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes
#4


import numpy as np
from biosppy.signals import ecg

# load raw ECG signal
signal = np.loadtxt('./examples/ecg.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True