import glob
import os
import pandas as pd
import numpy as np
import re
from string import digits
import random

d = {}
all = pd.DataFrame(columns=['Time', 'Sample #', 'Type', 'Sub', 'Chan', 'Num', 'PatID'])

for file in glob.glob(os.path.join(r'train', '*.txt')):
    f = pd.read_table(file, sep='\s{2,}')
    print file
    f.reset_index(level=0, inplace=True)
    f = f.rename(columns = {'index': 'Time', 'Time': 'Sample #', 'Sample #': 'Type', 'Type': 'Sub', 'Sub Chan': 'Chan', 'Num\tAux': 'Num'})
    k = file[-9:][:3]
    f['PatID'] = k
    all = pd.concat([all, f])

typ = ['N', 'R', 'L', 'V', 'A', '/']
train = pd.DataFrame(columns=['Time', 'Sample #', 'Type', 'Sub', 'Chan', 'Num', 'PatID'])

for t in typ:
    c = all[all['Type']==t]
    sample = c.sample(frac=0.7)
    train = pd.concat([train,sample])

print train.head()



from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  

import pickle
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl')
clf2 = joblib.load('filename.pkl')
clf2.predict(X[0:1])

