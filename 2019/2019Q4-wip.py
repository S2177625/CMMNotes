# find val of C, e and alpha ----------------------------------------------------------

from sympy import symbols, Eq, evalf, solve, sin, pi, rad

# Define symbols
C, e, a = symbols('C e a')

# Given constraints
theta_0 = 0
theta_minus_30 = -30
theta_30 = 30

# Express sine of angles in radians
sin_theta_minus_30 = sin(rad(theta_minus_30))
sin_theta_30 = sin(rad(theta_30))

# Given equations
eq1 = Eq(C / (1 + e * sin(a)), 6728)
eq2 = Eq(C / (1 + e * sin_theta_minus_30), 6870)
eq3 = Eq(C / (1 + e * sin_theta_30), 6615)

# Solve the system of equations
solution = solve((eq1, eq2, eq3), (C, e, a))

print(solution)

'''
gives solutions:
    C = 6059340/899
    e = 34/899
    a = 2.7231
'''
# sub in and minimize ----------------------------------------------------------

from scipy.optimize import minimize
import numpy as np

# Objective function to minimize (negative of V)
def eqs_R(theta):
    return (6059340/899)/(1 + (34/899) * np.sin(theta + 2.7231))


result = minimize(eqs_R, 0)


# Display the results
print(result)


    
