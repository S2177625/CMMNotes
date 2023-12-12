# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math


# initial conditions
x0 = 0
y0 = 1
x_final = 1 # total solution interval
h = 0.1# step size

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Initialize first element with initial conditions
y_eul[0] = y0
x_eul[0] = x0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    slope = model(y_eul[i],x_eul[i]) # compute the slope using the differential equation
    y_eul[i+1] = y_eul[i] + h * slope  # use the Euler method
