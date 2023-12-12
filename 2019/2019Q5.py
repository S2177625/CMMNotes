#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:14:41 2023

@author: ben
"""

from scipy.optimize import minimize
import numpy as np

# Objective function to minimize (negative of V)
def objective(theta):
    theta1, theta2, theta3 = theta
    return 24 * np.sin(theta1) + 30 * (1.2 * np.sin(theta1) + 1.5 * np.sin(theta2))

# Equality constraints
def eq_constraints(theta):
    theta1, theta2, theta3 = theta
    return [
        1.2 * np.cos(theta1) + 1.5 * np.cos(theta2) + np.cos(theta3) - 3.5,
        1.2 * np.sin(theta1) + 1.5 * np.sin(theta2) + np.sin(theta3)
    ]

# Initial guess
initial_guess = [0, 0, 0]

# Minimize the negative of V subject to the equality constraints
result = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': eq_constraints})

# Extract the optimal values of theta1, theta2, and theta3
theta_optimal = result.x

# Display the results
print("Optimal values:")
print("theta1 =", theta_optimal[0])
print("theta2 =", theta_optimal[1])
print("theta3 =", theta_optimal[2])
print("Minimum V value =", -result.fun)

