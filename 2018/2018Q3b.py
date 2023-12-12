#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:25:13 2023

@author: ben
"""

def synthetic_division(coefficients, root):
    n = len(coefficients) - 1
    quotient = [0] * n
    remainder = coefficients[0]

    for i in range(1, n):
        quotient[i - 1] = remainder
        remainder = coefficients[i] + root * remainder

    return quotient, remainder


def find_all_roots(coefficients):
    roots = []
    
    while len(coefficients) > 2:
        # Perform synthetic division to find a root
        root_candidate = 2  # You can change this initial guess
        _, remainder = synthetic_division(coefficients, root_candidate)
        
        # If the remainder is close to zero, consider it a root
        if abs(remainder) < 1e-10:
            roots.append(root_candidate)
            
            # Update coefficients for the next iteration
            coefficients, _ = synthetic_division(coefficients, root_candidate)
        else:
            break  # No more roots can be found

    # Solve the quadratic equation for the remaining coefficients
    quadratic_coefficients = coefficients[:3]
    roots += np.roots(quadratic_coefficients).tolist()

    return roots


# Coefficients of the polynomial: x^4 + 5x^3 + 15x^2 + 3x - 10
coefficients = [1, 5, 15, 3, -10]

# Find all roots
roots = find_all_roots(coefficients)

# Print the roots
print(f"All Roots: {roots}")

