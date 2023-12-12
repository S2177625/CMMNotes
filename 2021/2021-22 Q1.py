# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:14:41 2023

@author: Nick
"""

import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt 


f = lambda w: w**4 + 2*cm * w**3 + 3*sm * w**2 + cm*sm * w  + (sm)**2
cm = 12
sm =1500
coeff = [1, 2*cm, 3*sm, cm*sm, sm**2]


print("roots",np.roots(coeff))

'''
w = np.arange(-100, 100, 0.1)
fig = plt.figure(1)    
plt.plot(w, f(w), '.')
'''

#Q1a the code defines the coefficients and variables
#a numpy function is used to find the roots by computing the eigenvalues of the companion matrix

print()
wr = -0.62301963
wi = 24.03024141
w = complex(wr,wi)

#assuming j =1
aj = 0.1
phi = np.pi/8

#work = force*distance
x = lambda t: aj*np.exp(wr*t)*np.cos(wi*t+phi)
print("displacement at 10s:", x(10))
print("work:", x(10)*100 )

#Q1b work done = -0.006996828917452588