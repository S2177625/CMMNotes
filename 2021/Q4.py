import numpy as np
import math
import matplotlib.pyplot as plt

#Differential function
def diff(y,t):
    return Lambda*y + (1-Lambda)*np.cos(t) - (1+Lambda)*np.sin(t)

#Euler function, so that it can be run with different values of h
def Euler(h):
    #Defining time parameters and step count
    tf = 4*np.pi
    N = math.ceil(tf/h)
    t = np.linspace(0,tf,N)
    y = np.zeros(N)
    
    #Used to calculate error
    y_actual = np.sin(t) + np.cos(t)
    
    #Initial condition
    y[0] = 1
    
    #Performing Euler
    for i in range(N-1):
        dy = diff(y[i],t[i])
        y[i+1] = y[i] + dy*h
        
    error = abs(y_actual-y)
        
    return y, t, error

Lambda = -10

#Running Euler
y, t, error = Euler(0.01)

#Plotting
plt.plot(t,y)
plt.show()

#-------------------------------------------------------------------------
#PART A
#-------------------------------------------------------------------------
#Printing specific y values
print('At t = 2pi, y =',y[np.where(t==2*np.pi)])
print('At t = 4pi, y =',y[np.where(t==4*np.pi)])

#-------------------------------------------------------------------------
#PART B
#-------------------------------------------------------------------------
#Printing maximum error
print('Maximum error =', max(error))

#-------------------------------------------------------------------------
#PART C
#-------------------------------------------------------------------------
h = np.array([0.025,0.05,0.1,0.15,0.2,0.25,0.3])
error_2pi = np.zeros(len(h))
error_4pi = np.zeros(len(h))
for i in range(len(h)):
    y, t, error = Euler(h[i])
    #This min(range(len stuff gets the index of the value in t that is
    #the closest to a desired value, in this case 2pi and 4pi
    error_2pi[i] = error[min(range(len(t)), key=lambda i: abs(t[i] - 2*np.pi))]
    error_4pi[i] = error[min(range(len(t)), key=lambda i: abs(t[i] - 4*np.pi))]
    print('For h =', h[i], 'the error at 2pi is', error_2pi[i], 
          'and the error at 4pi is', error_4pi[i])