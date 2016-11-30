import matplotlib.pyplot as plt
import scipy.io 
import numpy as np
import obspy
from obspy.signal.detrend import polynomial
from statsmodels.tsa.ar_model import AR
import glob
import os
import pandas as pd
import numpy as np
import re
from string import digits
import random
from biosppy.signals import ecg
from sklearn import svm
import pywt
import pickle
from sklearn.externals import joblib
from obspy.signal.detrend import polynomial
import collections

BIHBase = 1024.
BIHGain = 200.
################################################################################################################################################################################################
# normalizing the signal - removing gain and base
#################################################################################################################################################################################################
def removeBaseGain(signal, base, gain):
    return (signal - base)/gain

################################################################################################################################################################################################
# getting the feature from heart beat signal
#################################################################################################################################################################################################
def getFeature(beat):
    feature = []
    cA, cD = pywt.dwt(beat, 'db8')
    cA, cD = pywt.dwt(cA, 'db8')
    cA, cD = pywt.dwt(cA, 'db8')
    cA, cD = pywt.dwt(cA, 'db8')
    #cA = polynomial(cA, order=12)
    feature.extend(cA)
    ar_mod = AR(beat)
    arm = ar_mod.fit(4)
    #plt.figure(1)
    beat = beat - np.mean(beat)
    #plt.plot(beat)
    #plt.figure(2)
    #plt.plot(arm.predict())
    #plt.figure(3)
    #plt.plot(cA)
    #plt.show()
    feature.extend(arm.params[0:4])
    return feature

#feature vector
x = []
sn = []
filen = []
#class vector
y = []
typ = ['N', 'R', 'L', 'V', 'A', '/']
dictBeats = {'N':1, 'L':2, 'R':3, '/':4, 'V':5, 'A':6}
invDictBeats = {v: k for k, v in dictBeats.items()}
################################################################################################################################################################################################
# Training the model
#################################################################################################################################################################################################
def trainModel():
    for file in glob.glob(os.path.join(r'train', '*.txt')):
        f = pd.read_table(file, sep='\s{2,}')
        print file
        f.reset_index(level=0, inplace=True)
        f = f.rename(columns = {'index': 'Time', 'Time': 'Sample #', 'Sample #': 'Type', 'Type': 'Sub', 'Sub Chan': 'Chan', 'Num\tAux': 'Num'})
        k = file[-9:][:3]
        f['PatID'] = k
        #all = pd.concat([all, f])
        fn = "train\\" + k + ".mat"
        signal = scipy.io.loadmat(fn)['val']
        signal = removeBaseGain(signal[0], BIHBase, BIHGain)
        for ind, sampleno in enumerate(f['Sample #']):
            if ((sampleno - 115) >= 0 and (650000 - sampleno) >= 185):
                #print "smp", sampleno
                if f['Type'][ind] == 'N' or f['Type'][ind] == 'R' or f['Type'][ind] == 'L' or f['Type'][ind] == 'V' or f['Type'][ind] == 'A' or f['Type'][ind] == '/':
                    y.append(dictBeats.get(f['Type'][ind]));
                    sn.append(sampleno)
                    filen.append(file)
                    #print fn, sampleno, f['Type'][ind] 
                    #print y
                    beat = signal[sampleno-115:sampleno+185]
                    feature = getFeature(beat)
                    plotCust(dictBeats.get(f['Type'][ind]), 1, beat)
                    plotCust(dictBeats.get(f['Type'][ind]), 2, feature)                    
                    x.append(list(feature))
  
 
################################################################################################################################################################################################
# Custom plotting
#################################################################################################################################################################################################                  
def plotCust(id, type, signal):
    plt.figure(id)
    if type == 1:
        plt.subplot(211)             # the first subplot in the first figure
        plt.xlabel('samples')        
        plt.ylabel('mv')
        plt.title('heart beat of type' + str(id))        
        plt.plot(signal)
    else:
        plt.subplot(212)
        plt.xlabel('samples')
        plt.ylabel('value')
        plt.title('feature (WT + AR) of type'+  str(id))        
        plt.plot(signal) 

################################################################################################################################################################################################
# parition the data from already read data
#################################################################################################################################################################################################
def partitionData():
    train = pd.DataFrame()
    test = pd.DataFrame()
    df = pd.DataFrame({'fn':filen, 'sn':sn, 'x':x, 'y':y})
    for t in typ:
        c = df[df['y']==dictBeats.get(t)]
        ta = np.random.rand(len(c)) < 0.7
        #sample = c.sample(frac=0.7)
        train = pd.concat([train,c[ta]])
        test = pd.concat([test, c[~ta]])
        #write remaining to file
    test.to_csv("test.csv")

################################################################################################################################################################################################
# Save model
#################################################################################################################################################################################################
def saveModel():
    clf = svm.SVC(C=4096, gamma=0.000244, kernel='rbf',decision_function_shape='ovo')
    clf.fit(train['x'].tolist(), train['y'].tolist())
    joblib.dump(clf, r'model\filename.pkl')

act = []
predicted = []
################################################################################################################################################################################################
# Testing the model from test partition
#################################################################################################################################################################################################
def testFromTestPartition():
    for index, beat in enumerate(test['x'].tolist()):
        #feature = getFeature(beat)
        #feature.shape
        pred = clf.predict([beat])
        #print test['fn'].tolist()[index],test['sn'].tolist()[index],"pred:",invDictBeats.get(pred[0]),"Actual:",invDictBeats.get(test['y'].tolist()[index])
        act.append(invDictBeats.get(test['y'].tolist()[index]))
        predicted.append(invDictBeats.get(pred[0]))

################################################################################################################################################################################################
# Loading the model and test files in memory
#################################################################################################################################################################################################
def getSignals():
    signals = {}
    find = 1
    for file in glob.glob(os.path.join(r'test', '*.mat')):
        f = pd.read_table(file, sep='\s{2,}')
        f.reset_index(level=0, inplace=True)
        f = f.rename(columns = {'index': 'Time', 'Time': 'Sample #', 'Sample #': 'Type', 'Type': 'Sub', 'Sub Chan': 'Chan', 'Num\tAux': 'Num'})
        k = file[-9:][:3]
        signal = scipy.io.loadmat(file)['val'][0]
        signal = removeBaseGain(signal, BIHBase, BIHGain)
        signals[find]=signal
        find = find + 1
    return signals


################################################################################################################################################################################################
# Testing the model from test folder
#################################################################################################################################################################################################
def testModel():
    clf2 = joblib.load(r'model\filename.pkl')
    for file in glob.glob(os.path.join(r'test', '*.mat')):
        f = pd.read_table(file, sep='\s{2,}')
        #print file
        f.reset_index(level=0, inplace=True)
        f = f.rename(columns = {'index': 'Time', 'Time': 'Sample #', 'Sample #': 'Type', 'Type': 'Sub', 'Sub Chan': 'Chan', 'Num\tAux': 'Num'})
        k = file[-9:][:3]
        #f['PatID'] = k
        #all = pd.concat([all, f])
        #fn = k + ".mat"
        signal = scipy.io.loadmat(file)['val']
        signal = removeBaseGain(signal[0], BIHBase, BIHGain)
        procSignal = ecg.ecg(signal=signal, sampling_rate=360., show=False)
        for sampleno in procSignal['rpeaks']:
            if ((sampleno - 115) >= 0 and (650000 - sampleno) >= 185):
                #feature extraction - wavelet transform and auto regressive modeling        
                beat = signal[sampleno-115:sampleno+185]
                feature = getFeature(beat)
                pred = clf2.predict([feature])
                print file,sampleno,invDictBeats.get(pred[0])
