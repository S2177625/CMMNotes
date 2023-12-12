# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
def model(y,t):
    L = -10
    dydt = L*y + (1-L)*np.cos(t) - (1+L)*np.sin(t)
    return dydt

# initial conditions
t0 = 0
y0 = 1
# total solution interval
t_final = 10

# Change step size at x=x2
t2 = 0.01
# step size for x<x2
h1 =  0.0002  
# increased step size for x>x2
h2 = 0.0021

h = h1

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(t2/h1) + math.ceil((t_final-t2)/h2)

#n_step = 16

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
t_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
t_eul[0] = t0 

# Populate the x array
for i in range(n_step):
    if t_eul[i]>t2:
        t_eul[i+1]  = t_eul[i]  + h2
    else:
        t_eul[i+1]  = t_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],t_eul[i]) 
    # change step after a certain x=x2
    if t_eul[i]>t2:
        h=h2
    else:
        h=h1
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope  
# ------------------------------------------------------
# plot results
plt.plot(t_eul, y_eul , 'b.-')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()
# ------------------------------------------------------
# Print the value of y(x) at t = 2pi
t_at_2pi_index = np.abs(t_eul - 2*np.pi).argmin()
y_at_2pi = y_eul[t_at_2pi_index]
print("y(2pi) =", y_at_2pi)
# ------------------------------------------------------
# Print the value of y(x) at t = 4pi
t_at_4pi_index = np.abs(t_eul - 4*np.pi).argmin()
y_at_4pi = y_eul[t_at_4pi_index]
print("y(4pi) =", y_at_4pi)


