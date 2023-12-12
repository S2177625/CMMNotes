##QUESTION 4

from sympy import Function, dsolve, Eq, Derivative, symbols

# Define symbols
t, m, b, k = symbols('t m b k')
x = Function('x')

# Define the differential equation for a mass-spring-damper system
diff_eq = Eq(m * Derivative(x(t), t, t) + b * Derivative(x(t), t) + k * x(t), 0)

# Solve the differential equation
solution = dsolve(diff_eq)

# Print the general solution
print("General Solution:")
print(solution)




from sympy import Function, dsolve, Eq, Derivative, symbols, cos, exp, sqrt
from sympy.abc import A, alpha, w

# Define symbols
t, m, b, k = symbols('t m b k')
x = Function('x')

# Given solution
A, alpha, w = symbols('A alpha w')
x_t = A * exp(-b/(2*m) * t) * cos(w * t + alpha)

# Define the relations
omega = sqrt(k/m)
damping_ratio = b/(2*m*omega)

# Substitute into the given solution
x_t = x_t.subs({w: omega, b/(2*m): damping_ratio})

# Differentiate x(t) with respect to t
dx_dt = x_t.diff(t)
dx2_dt2 = x_t.diff(t, 2)

# Substitute x(t), dx/dt, and d^2x/dt^2 into the dynamics equation
dynamics_eq = m * dx2_dt2 + b * dx_dt + k * x_t

# Simplify the result
dynamics_eq_simplified = dynamics_eq.simplify()

# Display the results
print("Given solution:")
print(x_t)

print("\nFirst derivative:")
print(dx_dt)

print("\nSecond derivative:")
print(dx2_dt2)

print("\nDynamics equation:")
print(dynamics_eq_simplified)








#equation of damped frequency is wd = sqrt(w0^2 - (b/2m)^2)

#theta(t) = A*exp(-bt/2m)
#time solved for reduction to 1% of original size of amplitude

##Only interested in amplitude, 0.01 * theta0 * exp(-b/2I * t) = theta0 * exp(-b/2I * (t + delta_t)

##by rearranging eqn: delta_t = 2m * ln(0.01)/0.1

b = 0.1
m = 1
delta_t = 2 * m *np.log(0.01)/-b
print(f'1% amplitude reached at {delta_t}')

##Solution matches up with desmos
##To halve the time simply use the same equation with b as unknown and time as 92.1034/2
time = delta_t/2
b = -2 * m *np.log(0.01)/time
print(f'damping ratio is {b}')


