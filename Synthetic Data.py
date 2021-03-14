# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:48:26 2021

@author: d-tje
"""

 

import numpy as np
import pandas as pd

x = np.random.randint(20,size=(1000,5))
xtest = np.random.randint(20,size=(100,5))
beta = np.random.randint(1,10,size=(5))
y = np.dot(x,beta)
ytest = np.dot(xtest,beta)


df = pd.DataFrame(data=x, index=None,columns=['x1','x2','x3','x4','x5'])
df['y'] = y

dftest = pd.DataFrame(data=xtest, index=None,columns=['x1','x2','x3','x4','x5'])
dftest['y'] = ytest

dfbeta = pd.DataFrame(data=beta)
dftest.to_csv("testdata.csv")
df.to_csv("trainingdata.csv")
dfbeta.to_csv("beta.csv")