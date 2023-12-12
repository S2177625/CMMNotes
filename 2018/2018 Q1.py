#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1A
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import numpy as np

#Defining the equation of the beam
def f(x):

    return (w/(120*E*I*L))*(-x**5 + 2*L**2*x**3 - L**4*x)

E=40000
L=800
I=40000
w=3.5

#Plotting the function to solve the maxima though inspection

# Plotting initial function and its quadratic approximation
x = np.linspace(0, 800, 1500)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='Target Function')
#ax.set_ylim([-4, 0])  if needed to determine the limit
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.grid()
plt.legend()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1B
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt

def gsection(ftn, xl, xm, xr, tol=0.01):  #Tolerance modified as per question

    # Calculate the golden ratio plus one
    gr1 = 1 + (1 + np.sqrt(5)) / 2

    # Initialize function values at bounds and midpoint
    '''Adjusted to match f(x)'''
    
    fl = f(xl)
    fr = f(xr)
    fm = f(xm)

    # Initialize previous midpoint for relative error calculation
    prev_xm = None

    # Iteration counter
    iteration = 0

    # Iteratively refine xl, xr, and xm
    while (xr - xl) > tol:
        iteration += 1  # Increment iteration counter

        if (xr - xm) > (xm - xl):
            y = xm + (xr - xm) / gr1
            fy = f(y)
            if fy <= fm:  # Changed condition for minimization
                xl, fl = xm, fm
                xm, fm = y, fy
            else:
                xr, fr = y, fy
        else:
            y = xm - (xm - xl) / gr1
            fy = f(y)
            if fy <= fm:  # Changed condition for minimization
                xr, fr = xm, fm
                xm, fm = y, fy
            else:
                xl, fl = y, fy

        # Calculate relative error if previous midpoint exists
        if prev_xm is not None:
            rel_error = abs((xm - prev_xm) / xm)

            # Print current iteration, values, and relative error
            print(f'Iteration {iteration}: xl = {xl}, xm = {xm}, xr = {xr}, f(xm) = {fm}, Relative Error = {rel_error}')

        prev_xm = xm

    return xm

# Set the bounds and midpoint for the golden section search
xl, xm, xr = 0, 400, 800  # Lower bound, initial midpoint guess, upper bound

# Perform the golden section search
x_min = gsection(f(x), xl, xm, xr, tol=1e-9)
print(f'x_min: {x_min}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1C
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from sympy import symbols, exp, series

# Define the symbols
x, h = symbols('x h')

# Define the function
f = exp(-x)

# Taylor series expansion around x=1 for the given perturbation h=0.1
taylor_expansion = series(f, x, 1, 4).removeO()
taylor_expansion_with_h = taylor_expansion.subs(x, 1 + h)

print('Without the perturbation the series is', taylor_expansion) 
print('With the perturbation the series is',taylor_expansion_with_h)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1D
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Redefine the function and its variables for clarity
x, h = symbols('x h')
f = exp(-x)

# Initial value of x and perturbation value
x0 = 1
h_value = 0.1

# Calculate the function value at the initial x and at the perturbed x
f_x0 = f.subs(x, x0)
f_x0_perturbed = f.subs(x, x0 + h_value)

# Calculate the change in the function value
delta_f = f_x0_perturbed - f_x0

# Calculate the sensitivity
sensitivity = delta_f / h_value

# Nicely formatted print statement
print(f"Value of f(x) at x = {x0}: {f_x0.evalf():.6f}")
print(f"Value of f(x) at x = {x0 + h_value}: {f_x0_perturbed.evalf():.6f}")
print(f"Change in f(x) due to perturbation: {delta_f.evalf():.6f}")
print(f"Sensitivity of f(x) to perturbation: {sensitivity.evalf():.6f}")

'''This question can also be solved by hand'''