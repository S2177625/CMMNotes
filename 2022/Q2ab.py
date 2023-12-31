import numpy as np
import matplotlib.pyplot as plt

def Euler(y0, h):
    #Differential function
    def diff(y):
        return 10*(y**2) - y**3
    
    #Defining step and time parameters
    t_final = 50
    N = int(t_final/h)
    
    #Assigining arrays for y and t values
    t = np.linspace(0,t_final,N+1)
    y = np.zeros(N+1)
    
    #Setting initial conditions
    y[0] = y0
    
    #something
    ignition = 0
    
    #Performing Euler
    for i in range(N):
        dydt = diff(y[i])
        y[i+1] = y[i] + h*dydt
        
        #When the gradient is large, a value for ignition delay can be calculated
        if dydt > 100:
            ignition = t[i]
    
    return t,y,ignition

t,y,ignition1 = Euler(0.02, 0.01)
_,_,ignition2 = Euler(0.01, 0.01)
_,_,ignition3 = Euler(0.005, 0.01)

#Plotting
plt.plot(t,y)
plt.show()

#Getting specific y values
print('At t = 4, y =',y[np.where(t==4)])
print('At t = 5, y =',y[np.where(t==5)])
print('At t = 10, y =',y[np.where(t==10)])

print('When y0=0.02, the ignition delay is',ignition1)
print('When y0=0.01, the ignition delay is',ignition2)
print('When y0=0.005, the ignition delay is',ignition3)

print(max(y))