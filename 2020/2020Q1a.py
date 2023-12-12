from sympy import symbols, diff, cosh, sqrt, Eq, solve, Function, init_printing
import math
import numpy as np

# Define the symbols
x, y, w, TA, y0 = symbols('x y w TA y0')

# Define the function y as a function of x
y_function = Function('y')(x)

# Define Equation 1B
equation_1B = Eq(y_function, TA/w * cosh(w/TA * x) + y0 - TA)

# Take the first derivative of y with respect to x
dy_dx = diff(y_function, x)

# Take the second derivative of y with respect to x
d2y_dx2 = diff(dy_dx, x)

# Substitute d2y/dx^2 into Equation 1A
equation_1A = Eq(d2y_dx2, w/TA * sqrt(1 + dy_dx**2))

# Simplify both sides of the equation
equation_1A_simplified = equation_1A.simplify()

# Check if the solution is True for all values of y0
solutions_for_y0 = solve(equation_1A_simplified, y0)

if all(equation_1A_simplified.subs(y0, sol).simplify() == True for sol in solutions_for_y0):
    print("Equation 1B is the general solution to Equation 1A.")
else:
    print("Equation 1B is not the general solution to Equation 1A.")
    
# part b ----------------


