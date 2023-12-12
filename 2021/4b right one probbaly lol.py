#2021/20211 4c
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# Functions that return dy/dx
# i.e., the equation we want to solve: dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(t, y):
    f_1 = -10 * y + 11 * np.cos(t) - (-9) * np.sin(t)
    return [f_1]

def exact_model(t):
    f_exact = np.sin(t) + np.cos(t)
    return f_exact
# ------------------------------------------------------

# ------------------------------------------------------
# Initial conditions
t0 = 0
y0 = 1

# Total solution interval
t_final = 4 * np.pi
# Step size
h = 0.1
h = 0.025
h = 0.05
h = 0.1
h = 0.15
h = 0.2
h = 0.25
 = 0.3
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# Number of steps
n_step = math.ceil(t_final / h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step + 1)
t_eul = np.zeros(n_step + 1)

# Initialize the first element of solution arrays
# with initial conditions
y_eul[0] = y0
t_eul[0] = t0

# Populate the x array
for i in range(n_step):
    t_eul[i + 1] = t_eul[i] + h

# Apply Euler method n_step times
for i in range(n_step):
    # Compute the slope using the differential equation
    [slope_1] = model(t_eul[i], y_eul[i])
    # Use the Euler method
    y_eul[i + 1] = y_eul[i] + h * slope_1
# ------------------------------------------------------

# ------------------------------------------------------
# Compute the exact solution for the same time points
y_exact = exact_model(t_eul)

# Compute true error at each time step
true_errors = np.abs(y_eul - y_exact)

# Report the maximum true error in the time range (0, 2*pi)
max_error_2pi = np.max(true_errors[:int(2 * np.pi / h)])

# Report the maximum true error in the time range (0, 4*pi)
max_error_4pi = np.max(true_errors)

print(f'Maximum true error in the time range (0, 2*pi): {max_error_2pi:.4f}')
print(f'Maximum true error in the time range (0, 4*pi): {max_error_4pi:.4f}')
# ------------------------------------------------------

# ------------------------------------------------------
# Plot results
plt.plot(t_eul, y_eul, 'b.-', label='Numerical Solution')
plt.plot(t_eul, y_exact, 'r-', label='Exact Solution')
plt.xlabel('t')
plt.ylabel('y(x)')
plt.legend()
plt.show()
# ------------------------------------------------------


#In this code, y_exact is computed using the exact_model function, and the true error at each time step is calculated as the absolute difference between y_eul and y_exact. The maximum true error is then determined using np.max in the specified time ranges.