# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:28:18 2018

@author: CAPTAIN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

data=pd.read_csv("abc.txt",names=["size","bedroom","price"])
data.dropna()
data=(data-data.mean())/(data.std())
x=np.array(data["size"]).reshape(-1,1)
x1=np.array(data["bedroom"]).reshape(-1,1)
y=np.array(data["price"]).reshape(-1,1)
one=np.ones((x.shape[0])).reshape(-1,1)
x=np.hstack((one,x,x1))

#theta=np.random.randn(3,1)*0.001
theta=np.array([1,2,3])
alpha=0.001

def gradient(x,y):
    global theta
    iter=10000
    for i in range(iter):
        yhat=np.matmul(x,theta)
        theta=theta-(alpha/len(x))*(np.matmul(x.T,(yhat-y)))
        c2=cost(x,y)
        if(iter%1000==0):
            print(c2)
    return theta

def cost(x,y):
    yhat=np.matmul(x,theta)
    c1=(1/(2*len(x)))*(np.sum(np.power(np.subtract(yhat,y),2)))
    return c1

gradient(x,y)
print(theta)

y1=np.array(x@theta)

