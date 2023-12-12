# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y

def euler(model, h, x0, y0, x_start, x_end, x_eul):

    h = 0.1
    # number of steps
    n_step = round((x_end-x_start)/h)
    # Definition of arrays to store the solution
    x_eul = np.linspace(x_start, x_end, round((x_end-x_start)/h))
    y_eul = np.linspace(0,       0,     round((x_end-x_start)/h))
    print("lenghts!", len(x_eul), len(y_eul))
    # Initialize first element of solution arrays 
    # with initial condition
    y_eul[0] = y0
    x_eul[0] = x0 

    # Apply Euler method n_step times
    for i in range(len(y_eul)-1):
        slope = model(y_eul[i],x_eul[i])    # compute the slope using the differential equation
        y_eul[i+1] = (y_eul[i] + h * slope) # use the Euler method
    
    return x_eul, y_eul
