# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:25:48 2023

@author: Nick
"""
import numpy as np 
from matplotlib import pyplot as plt 

ao = 0.05
b = 0.1
m = 1
wo = 5
phi = 0

w = np.sqrt(wo**2-(b/(2*m))**2)
t = np.arange(0, 100, 0.1)
x = np.arange(0, 100, 0.1)
#x = np.linspace(0,999,1000)

for i in range(1000):
    x[i] = ao * np.exp(-(b*t[i]/(2*m))) * np.cos(wo*t[i]+phi)

fig = plt.figure(1)    
plt.plot(t[700:], x[700:], '-')
plt.axhline(y=0.01*ao)


# Q4c
# by inspection of the graph and changing plotting boundaries
# the time taken to reach 0.01*Ao is 92s
# this could also be found by reading x where t is a multiple of 2pi/5 and checking...

# Q4d
# b = 0.2 is required to half the time for the amplitude to reach 0.01*Ao


    