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

# List of step sizes to assess
step_sizes = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Loop over different step sizes
for h in step_sizes:
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

    print(f'Step size h = {h:.3f}')
    print(f'Maximum true error in the time range (0, 2*pi): {max_error_2pi:.4f}')
    print(f'Maximum true error in the time range (0, 4*pi): {max_error_4pi:.4f}')



#as h decreases, error will decrease: convergense of numerical method
#smaller h values = more steps in time range

#as h becomes small, reduction will slow down