#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:18:29 2023

@author: ben
"""

from scipy.optimize import minimize_scalar

# Define the function
def objective_function(x, w, L, E, I):
    return -(w / (120 * E * I * L)) * x**5 + (w / (120 * E * I * L)) * 2 * L**2 * x**3 - (w / (120 * E * I * L)) * L**4 * x

# Given parameters
w = 3.5
L = 800
E = 40000
I = 40000

# Set the initial guesses for the lower and upper bounds
x_lower = 0
x_upper = L

# Set the tolerance for stopping criterion (approximate error)
epsilon_s = 0.01  # 1% error

# Perform optimization
result = minimize_scalar(objective_function, bounds=(x_lower, x_upper), args=(w, L, E, I), method='bounded', tol=epsilon_s)

# Extract the result
optimized_x = result.x
optimized_value = result.fun

# Print the result
print(f"Optimized x: {optimized_x}")
print(f"Optimized function value: {optimized_value}")
