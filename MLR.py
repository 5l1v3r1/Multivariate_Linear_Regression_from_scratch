# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:28:18 2018

@author: CAPTAIN     
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


x,y,alpha,theta,data,one=None,None,None,None,None,None

def data_preprocess():
    global x,y,one,data,theta,alpha
    data=pd.read_csv("abc.txt",names=["size","bedroom","price"])
    data.dropna()
    data=(data-data.mean())/(data.std())
    x=np.array(data.iloc[:,0:2])
    y=np.array(data["price"])
    theta=np.random.randn(3,1)
    alpha=0.01
    y=y.reshape(-1,1)
    one=np.ones((x.shape[0])).reshape(-1,1)
    x=np.hstack((one,x))


def gradient(x,y):
    global theta
    iter=1000
    for i in range(iter):
        yhat=np.matmul(x,theta)
        theta=theta-(alpha/len(x))*(np.matmul(x.T,(yhat-y)))
        c2=cost(x,y)
        if(iter%100==0):
            print(c2)
    return theta

def cost(x,y):
    yhat=np.matmul(x,theta)
    c1=(1/(2*len(x)))*(np.sum(np.power(np.subtract(yhat,y),2)))
    return c1

data_preprocess()
gradient(x,y)
print(theta)

y1=np.array(x@theta)

