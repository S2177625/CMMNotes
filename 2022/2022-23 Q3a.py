# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:41:19 2023

@author: Nick
"""
import numpy as np
from matplotlib import pyplot as plt 

#by solving a 100 iterations of the function for different initial
#inputs it can be observed the function eventually tends to 
#different values where xn = xn+1

#we can define two arrays to store the initial and final outputs

x_initial = np.arange(0.01,np.pi,0.01)
x_final = np.arange(0.01,np.pi,0.01)

#we can run a loop for the size of the array and for each loop
#run 100 iterations for the initial value in the array
#and store the final value from that iteration

for i in range(x_initial.size):
    x_value = x_initial[i]
    for j in range(5):
        x_value = 1/np.sin(x_value) + 1/4
    x_final[i] = x_value
    print(i,x_initial[i],x_final[i])

# plots the range of final iterations for initial inputs
#we can observe the iterative function tends to two values
#around -0.9 and 1.3
plt.plot(x_initial, x_final, '.')

#Q3a
#the value of x in the domain is 1.29



#code to solve single iteration
'''
for i in range(100):
    #print(np.sin(y[i]))
    y.append(1/np.sin(y[i]) + 1/4)
    #print((1/np.sin(1.5*np.pi/180) + 1/4))
        
    x.append(x[i]+1)
    print(x[i],y[i])

fig = plt.figure(1)    
plt.plot(x, y, '.')
plt.xlabel('x')
plt.ylabel('y=sin(x)')
'''