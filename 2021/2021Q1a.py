#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:00:31 2023

@author: ben
"""
import numpy as np

# Given values
c_over_m = 12  # s^-1
s_over_m = 1500  # s^-2

# Coefficients for the fourth-order polynomial equation
coefficients = [1, 2 * c_over_m, 3 * s_over_m, c_over_m * s_over_m, s_over_m**2]

# Calculate the roots using numpy.roots()
roots = np.roots(coefficients)

# Extract real and imaginary parts of the roots
omega_r = np.real(roots)
omega_i = np.imag(roots)

# Print the results
for i in range(len(roots)):
    print("Root {}: {:.2f} + {:.2f}i".format(i + 1, omega_r[i], omega_i[i]))

'''
.2f part rounds to 2 decimal points, format, inputs that part into the corresposnign bracket

np.roots can solve for complex roots if extracted.
'''
