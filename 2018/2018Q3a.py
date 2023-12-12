#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:43:57 2023

@author: ben
"""

import cmath  # Importing the complex math module for handling complex roots

# Coefficients of the quadratic equation
a = 1
b = 10
c = 25

# Calculate the discriminant
discriminant = b**2 - 4*a*c

# Check if the discriminant is non-negative for real roots
if discriminant >= 0:
    # Calculate the two real roots
    root1 = (-b + (discriminant)**0.5) / (2*a)
    root2 = (-b - (discriminant)**0.5) / (2*a)
    print(f"Real Roots: {root1}, {root2}")

else:
    # Calculate the two complex roots
    root1 = (-b + cmath.sqrt(discriminant)) / (2*a)
    root2 = (-b - cmath.sqrt(discriminant)) / (2*a)
    print(f"Complex Roots: {root1}, {root2}")
    