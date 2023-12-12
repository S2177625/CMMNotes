# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:41:58 2023

@author: Ben
"""

from sympy import symbols, diff, cos, exp, sqrt, Eq, solve, Function, init_printing
import math
import numpy as np

A0, b, m, t, w, k , phi  = symbols('A0 b m t w k phi')

#define eq 4.1 and differentials
x_function = Function('x')(t)
equation_42 = Eq(x_function, A0*exp(-b*t/2*m)*cos(w*t+phi))
dx_dt = diff(x_function, t)
d2x_dt2 = diff(dx_dt, t)

#solve and simplify for eq 4.1
equation_41 = Eq(0,  m*d2x_dt2 + b*dx_dt + k*equation_42.rhs)
equation_41_simplified = equation_41.simplify()

#solve eq 4.1 for Ao
solutions_for_A0 = solve(equation_41_simplified, A0)
print("eq 4.1",equation_41_simplified)
print()
print("solutions for Ao",solutions_for_A0)

#somehow compares and checks if it's the general solution
if all(equation_41_simplified.subs(A0, sol).simplify() == True for sol in solutions_for_A0):
    print("Equation 1B is the general solution to Equation 1A.")
else:
    print("Equation 1B is not the general solution to Equation 1A.")