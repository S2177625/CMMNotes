#22/23 question 2(a)

import numpy as np
import matplotlib.pyplot as plt
import math

def model(y, t):
    dydt = 10 * y**2 - y**3
    return dydt

# Initial conditions
t0 = 0
y0 = 0.02

t_final = 10
# Step size (smaller step size can lead to better accuracy)
h = 0.01

# Number of steps
n_step = math.ceil(t_final / h)

# Definition of arrays to store solution
y_eul = np.zeros(n_step + 1)
t_eul = np.zeros(n_step + 1)

# Initialize first element of solution arrays with initial conditions
y_eul[0] = y0
t_eul[0] = t0

# Populate the x array
for i in range(n_step):
    t_eul[i + 1] = t_eul[i] + h

# Applying Euler method
for i in range(n_step):
    # Compute the slope using the differential equation
    slope = model(y_eul[i], t_eul[i])
    # Using Euler method
    y_eul[i + 1] = y_eul[i] + h * slope

###################################

# Print values at specific times
t_values = [4, 5, 10]

print('Time (t)   Value of y')
for t_val in t_values:
    index = int(t_val / h)
    print(f"{t_val}        {y_eul[index]}")
print(f"Selected step size h: {h}")

