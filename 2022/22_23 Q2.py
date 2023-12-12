##QUESTION 2: Euler method approximation

import numpy as np
import math
import matplotlib.pyplot as plt

def model(y, t):
    dydt = 10*y**2 - y**3
    return dydt

t0 = 0
y0 = 0.02
t_final = 20
h = 0.001
n_steps = math.ceil(t_final/h)
y_eul = np.zeros(n_steps + 1)
t_eul = np.zeros(n_steps + 1)
y_eul[0] = y0
t_eul[0] = t0

for i in range(n_steps):
    t_eul[i+1] = t_eul[i] + h
    slope = model(y_eul[i], t_eul[i])
    y_eul[i+1] = y_eul[i] + h * slope

# Print values at specific time points
times_of_interest = [4.9, 5, 5.1]
for time_point in times_of_interest:
    index_at_time_point = int(time_point / h)
    print(f"At t = {time_point}: y = {y_eul[index_at_time_point]}")

# Plot the results
plt.plot(t_eul, y_eul, 'b.-')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()

## The ignition delay for initial conditions of 0.02, 0.01 and 0.005 are roughly 5s, 10s and 20s respectively

##From experimentation with step values, system starts to becomes unstable for h > 0.02
'''
##Iplicit euler method is unconditionally stable
##Implicit methods employ the use fo locations that havent been computed yet

**Stability of Numerical Methods for ODEs:**

The stability of numerical methods for solving Ordinary Differential Equations (ODEs) is a crucial consideration in ensuring the accuracy and reliability of the solutions. A stable method produces solutions that do not exhibit unbounded growth or oscillations as the numerical integration progresses. Stability is particularly important in the context of ODEs because it directly impacts the reliability of predictions and the ability to capture the behavior of the underlying physical system.

**Explicit vs. Implicit Euler Methods:**

1. **Explicit Euler Method:**
   - In the explicit Euler method, the solution at the next time step is computed solely based on the
   information from the current time step.
   - The update formula is `y_{i+1} = y_i + h * f(y_i, t_i)`, where `h` is the step size, `f` is the
   derivative function, and `y_i` and `t_i` are the solution and time at the current step.
   - Explicit methods are easy to implement but may suffer from stability issues, especially when the 
   step size is large.

2. **Implicit Euler Method:**
   - In the implicit Euler method, the solution at the next time step is obtained by solving an equation
   that involves both the current and next time steps.
   - The update formula is `y_{i+1} = y_i + h * f(y_{i+1}, t_{i+1})`, where the function `f` is evaluated
   at the next time step values.
   - Implicit methods can be more stable for certain types of ODEs, as they inherently account for
   future behavior. However, they often require solving nonlinear equations at each time step, which
   can be computationally more demanding.

**Implications for Stability and Step Size (h):**

1. **Explicit Methods:**
   - Explicit methods are conditionally stable, meaning that there is a maximum allowable step size for 
   stability.
   - The stability is often characterized by the Courant-Friedrichs-Lewy (CFL) condition, which imposes
   a restriction on the step size based on the properties of the system being modeled.
   - Large step sizes may lead to instability and numerical artifacts.

2. **Implicit Methods:**
   - Implicit methods are unconditionally stable for certain types of problems, allowing for larger
   step sizes without stability concerns.
   - However, the computational cost associated with solving implicit equations at each step can be
   higher.

**Choice of Step Size (h):**
   - The selection of the step size (h) is a trade-off between accuracy and stability.
   - Smaller step sizes generally improve accuracy but increase computational cost.
   - The stability of explicit methods is more sensitive to the choice of step size, requiring careful
   consideration to prevent instability.
   - Implicit methods often offer more flexibility in choosing step sizes, allowing for larger steps
   without sacrificing stability.

In summary, the stability of numerical methods for ODEs is a critical factor in their effectiveness.
 The explicit and implicit Euler methods illustrate different approaches, with explicit methods
 requiring careful consideration of step size for stability and implicit methods offering more 
 stability flexibility at the cost of increased computational complexity. The choice of step size
 should be made based on a balance between stability requirements and computational efficiency.

'''



