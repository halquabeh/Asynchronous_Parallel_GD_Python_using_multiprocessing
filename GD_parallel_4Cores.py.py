#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:04:11 2021
Gradient Descent with four cores in parallel
@author: halqu
"""
from numpy import *
from pandas import *
import random as random
from time import perf_counter
from numpy import *
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
training_original = read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data',header = None,delim_whitespace = True)
test_original = read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test',header = None,delim_whitespace = True)
training_original = training_original.iloc[:,0:23].values
training=delete(training_original,2,axis=1)
test_original = test_original.iloc[:,0:23].values
test = delete(test_original,2,axis=1)
m,n = shape(training)
# deleting samples with missing label '?'
for i in range(m):
        if training[i,-1] =='?':
            delete(training,i,axis=0)
# Replacing missing features '?' data with 0.0.
for i in range(m):
    for j in range(n):
         if training[i,j] =='?':
                training[i,j] = 0
# Repalcing labels values 2.0 and 3.0 with 0.0;
for i in range(m):
    for j in range(n):
        training[i,j] =float(training[i,j])
    if training[i,-1]!=1.0:
        training[i,-1]=0.0
#training = array(training)
training=array(training,dtype=float32)
# Do the same for the test Data
m,n = shape(test)
for i in range(m):
    if test[i,-1] =='?':
        delete(test,i,axis=0)
for i in range(m):
    for j in range(n):
         if test[i,j] =='?':
                test[i,j] = 0
for i in range(m):
    for j in range(n):
        test[i,j] =float(test[i,j])
    if test[i,-1]!=1.0:
        test[i,-1]=0.0
test = array(test,dtype=float32)

#Seperate data from labrls
X_train = training[:,0:21-1]
y_train = training[:,[22-1]]
m,n=shape(X_train)
# THe hypothesis is h(X,theta)
w = zeros((n,1))
b= 0
h=dot(X_train,w) + b
# or we Can make the weights together in big matrix as:
X_train = concatenate((X_train,ones((m,1))),axis=1)
m,n=shape(X_train)
W = zeros((n,1))

def sigmoid(inx):
         return 1.0/(1+exp(-inx))
# The hypothesis is now
def H(W,X_train):
    h=dot(X_train,W)
    H = list(zip(*[sigmoid(x) for x in h]))
    H = array(H);H = H.T
    return H
# The prediction bassed on H
def pred(H):
    P = zeros((m,1))
    for i in range(0,m):
        if H[i]>=0.5:
            P[i] = 1.0
        else : P[i] = 0.0
    return P

def GD(X_train,y_train,alpha,Ws,iterations,idd):
    m,n = shape(X_train)
    W = array(Ws);W=W.reshape((n,1)) # read model from shared memory
    start_time=time.time()
    L=[] # list to monitor losses 
    T=[] # amd time
    for i in range(0,iterations):
        dL = (dot(X_train.T,(H(W,X_train)-y_train)))*1/m
        W = W - alpha * dL
        if (i%20==0):
            l=sum(square(H(W,X_train)-y_train))
            L.append(l)
            T.append(time.time()-start_time)
        for i in range(n):
             Ws[i]=W[i,:]   # write back the updated model to shared memory
        W = array(Ws);W=W.reshape((n,1)) # and read last update
    np.savez(str(idd)+'four.npz', loss1=L, time1=T)
    return Ws

# The main process shown be active
if __name__=='__main__':
    W = mp.Array('d',n) # Assign a shared array for the model
    #W=zeros((n,1))
    mid=int(m/2)  # Mid of data samples
    qua1=int(m/4) # quarter
    qua2=mid+qua1 # three quarters
    iterations=10000 
    alpha=0.0001
    # The process class is used from mutli processing to divide the data on 
    # multi cores, every process copies the same function (GD) to it own memory
    # and perform the updates, however the model is shared between all processes
    p1=mp.Process(target=GD,args=(X_train[:qua1,:],y_train[:qua1,:],alpha,W,iterations,1))
    p2=mp.Process(target=GD,args=(X_train[qua1:mid,:],y_train[qua1:mid,:],alpha,W,iterations,2))
    p3=mp.Process(target=GD,args=(X_train[mid:qua2,:],y_train[mid:qua2,:],alpha,W,iterations,3))
    p4=mp.Process(target=GD,args=(X_train[qua2:-1,:],y_train[qua2:-1,:],alpha,W,iterations,4))
    start_time = time.time()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    end_time=time.time()
    # Convert the shared model to numpy array and calculate the final accuracy
    W = array(W)
    W=W.reshape((n,1))
    H=H(W,X_train)
    l = sum(square(H-y_train))
    print(f'The loss is {l} and number of True Preditcs is {sum(y_train==pred(H))}')
    print('Elapsed Ttme = %g seconds.' % (end_time-start_time) )

