import math

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             3A
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''Finding a values of x which the iterative technique converges to 0'''


def fixed_point_iteration(g, x0, tolerance=1e-6, max_iterations=1000):

    xn = x0
    for _ in range(max_iterations):
        x_next = g(xn)
        
        # Check if the absolute difference is within the tolerance
        if abs(x_next - xn) < tolerance:
            return x_next
        
        xn = x_next
    
    return xn  # Return the approximation if max iterations reached

# Define the iterative function g(x)
g = lambda x: 1 / math.sin(x) + 1/4

# Initial guess within the domain 0 < x < 4
x0 = 2

# Apply the Fixed Point Iteration method
fixed_point = fixed_point_iteration(g, x0)

print('3A', fixed_point)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             3B - Bisection Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bisection_method(f, a, b, tolerance=1e-6, max_iterations=1000):

    for i in range(max_iterations):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < tolerance or (b - a) / 2 < tolerance:
            return m, i, abs(fm)  # Return the value, the number of iterations, and the absolute error
        
        if f(a) * fm < 0:
            b = m
        else:
            a = m

    return (a + b) / 2, max_iterations, abs(f((a + b) / 2))  # Return the approximation if max iterations reached

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             3C - Secant Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def secant_method(f, x0, x1, tolerance=1e-6, max_iterations=1000):

    for i in range(max_iterations):
        if f(x1) == f(x0):
            return x1, i, 0  # If consecutive function values are equal, return current value as root
        
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        
        if abs(x2 - x1) < tolerance:
            return x2, i, abs(f(x2))  # Return the value, the number of iterations, and the absolute error
        
        x0, x1 = x1, x2
        
    return x1, max_iterations, abs(f(x1))  # Return the approximation if max iterations reached

# Define the function
f = lambda x: 1 / math.sin(x) + 1/4 - x

# Apply the Bisection Method
root_bisection, iterations_bisection, error_bisection = bisection_method(f, 1, 2)

# Apply the Secant Method
root_secant, iterations_secant, error_secant = secant_method(f, 1, 2)

# Print the results in a formatted way
print(f'3B (Bisection Method): Root = {root_bisection}, Iterations = {iterations_bisection}, Absolute Error = {error_bisection}')
print(f'3C (Secant Method): Root = {root_secant}, Iterations = {iterations_secant}, Absolute Error = {error_secant}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             3D
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''This analysis indicates that the Secant Method reached a solution that is closer to the true root faster than the Bisection Method in terms of the number of iterations. However, it's important to note that the absolute error is not the true error (which would be the difference between the computed root and the true root), but rather the difference in the function values at the final iteration, which serves as a proxy for the error in this case.

The Secant Method provided a more accurate root with fewer iterations because it uses a form of numerical differentiation, which, under the right circumstances, can provide faster convergence than the Bisection Method, which is a bracketing method and converges linearly.'''



