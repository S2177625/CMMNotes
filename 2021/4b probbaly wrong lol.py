import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# Function that returns dy/dx
# i.e. the equation we want to solve: dy/dx = f(t, y)
def model(t, y):
    f_1 = -10 * y + 11 * np.cos(t) - (-9) * np.sin(t)
    return f_1

# Initial conditions
t0 = 0
y0 = 1

# Total solution interval
t_final = 4 * np.pi

# Step size
h = 0.1

# Number of steps
n_step = math.ceil(t_final / h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step + 1)
t_eul = np.zeros(n_step + 1)

# Initialize first element of solution arrays with initial condition
y_eul[0] = y0
t_eul[0] = t0

# Populate the t array
for i in range(n_step):
    t_eul[i + 1] = t_eul[i] + h

# Apply Euler method n_step times
for i in range(n_step):
    # Use the Euler method
    y_eul[i + 1] = y_eul[i] + h * model(t_eul[i], y_eul[i])

# Solutions at t=2pi and t=4pi
y_2pi = y_eul[int(2 * np.pi / h)]
y_4pi = y_eul[-1]

print("The solution of y at t=2*pi is", y_2pi)
print("The solution of y at t=4*pi is", y_4pi)
# ------------------------------------------------------

def exact_model(t):
    f_exact = np.sin(t) + np.cos(t)
    return f_exact

# Definition of arrays to store the exact solution
exacty_eul = np.zeros(n_step + 1)

# Initialize first element of exact solution arrays with initial condition
exacty_eul[0] = y0

# Apply Euler method n_step times for the exact solution
for i in range(n_step):
    # Use the Euler method for the exact solution
    exacty_eul[i + 1] = exacty_eul[i] + h * exact_model(t_eul[i])

    
    
exacty_2pi = exacty_eul[int(2 * np.pi / h)]    
exacty_4pi = exacty_eul[-1]
    
print("The exact solution of y at t=2*pi is", exacty_2pi)
print("The exact solution of y at t=4*pi is", exacty_4pi)


true_errors2pi = np.abs(y_2pi - exacty_2pi)
true_errors4pi = np.abs(y_4pi - exacty_4pi)

print("Maximum true error in the time range (0, 2*pi)", true_errors2pi)
print("Maximum true error in the time range (0, 4*pi)", true_errors4pi)

    

