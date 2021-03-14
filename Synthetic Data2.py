# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:48:53 2021

@author: d-tje
"""


import numpy as np
import pandas as pd

x = np.random.randint(20,size=(1000,5))
x2 = np.ones([1000,5])
x2[:,1] = x[:,1]
x2 = np.multiply(x,x2)


xtest = np.random.randint(20,size=(100,5))
x2test = np.ones([100,5])
x2test[:,1] = xtest[:,1]
x2test = np.multiply(xtest,x2test)


beta = np.random.randint(1,10,size=(5))
y = np.dot(x2,beta)
ytest = np.dot(x2test,beta)


df = pd.DataFrame(data=x, index=None,columns=['x1','x2','x3','x4','x5'])
df['y'] = y

dftest = pd.DataFrame(data=xtest, index=None,columns=['x1','x2','x3','x4','x5'])
dftest['y'] = ytest

dfbeta = pd.DataFrame(data=beta)
dftest.to_csv("testdata2.csv")
df.to_csv("trainingdata2.csv")
dfbeta.to_csv("beta2.csv")