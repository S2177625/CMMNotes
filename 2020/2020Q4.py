#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:06:24 2023

@author: ben
"""

'''
since u, m0, q dont change with regard to time, simple integration of equation will suffice
'''


from sympy import symbols, integrate, ln

# Define symbols
t, u, m0, q = symbols('t u m0 q')

# Given values
u_val = 1800
m0_val = 160000
q_val = 2500

# Define the expression to be integrated
v = u * ln(m0 / (m0 - q * t))

# Integrate with respect to t from 0 to 30
result = integrate(v, (t, 0, 30))

# Substitute the given values into the result
result_evaluated = result.subs({u: u_val, m0: m0_val, q: q_val}).evalf() 

print("The result of the integration is:", result_evaluated)
