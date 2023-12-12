#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:14:42 2023

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import math
# ------------------------------------------------------
# Function that returns dy/dx for the given differential equations
def model(theta, omega):
    g = 9.8  # acceleration due to gravity
    l = 1.0  # length of the pendulum
    dtheta_dt = omega
    domega_dt = -g / l * np.sin(theta)
    return [dtheta_dt, domega_dt]
# ------------------------------------------------------
# Initial conditions
theta0 = np.pi / 4  # initial angle
omega0 = 0.0        # initial angular velocity
# ------------------------------------------------------
# Total solution interval
t_final = 20
# Step size
h = 0.001
# ------------------------------------------------------
# Euler method
n_steps = math.ceil(t_final / h)
# ------------------------------------------------------
# Arrays to store the solution
theta_euler = np.zeros(n_steps + 1)
omega_euler = np.zeros(n_steps + 1)
t_euler = np.zeros(n_steps + 1)
# ------------------------------------------------------
# Initialize first element of solution arrays with initial conditions
theta_euler[0] = theta0
omega_euler[0] = omega0
t_euler[0] = 0.0
# ------------------------------------------------------
# Apply Euler method n_steps times
for i in range(n_steps):
    # Compute the slope using the differential equation
    [slope_theta, slope_omega] = model(theta_euler[i], omega_euler[i])
    # Use the Euler method
    theta_euler[i + 1] = theta_euler[i] + h * slope_theta
    omega_euler[i + 1] = omega_euler[i] + h * slope_omega
    t_euler[i + 1] = t_euler[i] + h
# ------------------------------------------------------
# Plot results
plt.plot(t_euler, theta_euler, 'b.-', label='Theta(t)')
plt.plot(t_euler, omega_euler, 'r.-', label='Omega(t)')
plt.xlabel('Time (t)')
plt.ylabel('Theta(t), Omega(t)')
plt.legend()
plt.show()
# ------------------------------------------------------
#Print the value of y(x) at x = 10
t_at_10_index = np.abs(t_euler - 10).argmin()
theta_at_10 = theta_euler[t_at_10_index]
omega_at_10 = omega_euler[t_at_10_index]
print("omega(10) =", omega_at_10)
print("theta(10) =", theta_at_10)
# ------------------------------------------------------
#Print the value of y(x) at x = 20
t_at_20_index = np.abs(t_euler - 20).argmin()
theta_at_20 = theta_euler[t_at_20_index]
omega_at_20 = omega_euler[t_at_20_index]
print("omega(20) =", omega_at_20)
print("theta(20) =", theta_at_20)
# ------------------------------------------------------
#Print the value of y(x) at x = 30
t_at_30_index = np.abs(t_euler - 30).argmin()
theta_at_30 = theta_euler[t_at_30_index]
omega_at_30 = omega_euler[t_at_30_index]
print("omega(30) =", omega_at_30)
print("theta(30) =", theta_at_30)

# ------------------------------------------------------
'''
part b, measure graph manually

part c, explain

part d, skip
'''
