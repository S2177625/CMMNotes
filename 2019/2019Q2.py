#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:49:21 2023

@author: ben
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:29:32 2023

@author: emily
"""

#-----------------------

# 2019 QUESTION 2

#-----------------------

#Minimising integrals with coefficient p


import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, integrate, solve, pi

x, p = symbols("x p")
function = sin(x) * cos(p * x)
function_i = integrate(function, (x, 0, pi))

lengthGraph = 10
n_array = np.zeros(lengthGraph)
substitution = np.zeros(lengthGraph)

for p_val in range(1, lengthGraph + 1):  # Avoid division by zero when p = 0
    n_array[p_val - 1] = p_val
    substitution[p_val - 1] = function_i.subs({p: p_val})

    # Check if the integral is 0
    if substitution[p_val - 1] == 0:
        print("The value of integration is 0 at p value ", p_val)

        # Solve the equation for the specific p value
        answer = solve(function.subs({p: p_val}), x)
        print("Solution for x:", answer)

plt.figure()
plt.title("Integration of sin(x)*cos(p*x) with respect to x")
plt.plot(n_array, substitution, '-o')
plt.xlabel("p values")
plt.ylabel("Function value")
plt.show()
