import numpy as np
from scipy.misc import derivative

# Define the function
f = lambda x: np.exp(-x)

# Initial value and perturbation
x_initial = 1
h = 0.1

# Calculate the change in the value of f(x)
change_in_value = f(x_initial + h) - f(x_initial)

# Calculate the sensitivity
# Sensitivity is the derivative of f at x_initial times the perturbation
sensitivity = derivative(f, x_initial, dx=1e-6) * h

# Print the results with six significant decimals
print('Change in value of f(x): {:.6f}'.format(change_in_value))
print('Sensitivity of f(x): {:.6f}'.format(sensitivity))
