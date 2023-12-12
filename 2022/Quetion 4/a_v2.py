from sympy import symbols, Eq, Function, dsolve, diff, cos, exp
from sympy.abc import t

# Define the symbols
m, b, k, omega, omega_0, phi, A_0 = symbols('m b k omega omega_0 phi A_0')

# Define the function x(t)
x = Function('x')(t)

# Differential equation (Equation Q4.1)
differential_eq = Eq(m * x.diff(t, 2) + b * x.diff(t) + k * x, 0)

# General solution provided (Equation Q4.2)
general_solution = A_0 * exp(-b*t/(2*m)) * cos(omega*t + phi)

# Calculate omega from the given equation (Equation Q4.3)
omega = (omega_0**2 - (b/(2*m))**2)**0.5

# Substitute omega in general solution
general_solution = general_solution.subs(omega, (omega_0**2 - (b/(2*m))**2)**0.5)

# Now, let's solve the differential equation using dsolve from sympy
solved_eq = dsolve(differential_eq, x)

# Display the general solution from sympy
print(solved_eq, general_solution)


"""
Have a look at the bisection method.

Just splitting it into order odes. 

3a. Look at it to understand the logic.c

"""