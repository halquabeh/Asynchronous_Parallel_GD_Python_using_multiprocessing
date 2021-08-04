# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:05:26 2021

@author: halqu
"""

import numpy as np
import matplotlib.pyplot as plt

oneCore=np.load('onecore.npz')
loss_one=oneCore['loss']
time_one = oneCore['time']
L1=loss_one
T1=time_one
print(L1[-1])
print(T1[-1])
twoCore1=np.load('1two1.npz')
loss1=twoCore1['loss1']
time1 = twoCore1['time1']
twoCore2=np.load('2two1.npz')
loss2=twoCore2['loss1']
time2 = twoCore2['time1']
L2=loss1+loss2
T2=time1
print(L2[-1])
print(T2[-1])

fourcore1=np.load('1four.npz')
loss1=fourcore1['loss1']
time1 = fourcore1['time1']
fourcore2=np.load('2four.npz')
loss2=fourcore2['loss1']
time2 = fourcore2['time1']
fourcore3=np.load('3four.npz')
loss3=fourcore3['loss1']
time3 = fourcore3['time1']
fourcore4=np.load('4four.npz')
loss4=fourcore4['loss1']
time4 = fourcore4['time1']
L3=loss1+loss2+loss3+loss4
T3=time1
print(L3[-1])
print(T3[-1])

fig, ax = plt.subplots()
ax.plot(T1,L1,'-.k',label='Sequential')
ax.plot(T2,L2,':r',label='Dual Cores')
ax.plot(T3,L3,'--c',label='Quad Cores')
leg=ax.legend();
plt.xlim(0,14)
plt.xlabel('Time [s]')
plt.ylabel('Loss')