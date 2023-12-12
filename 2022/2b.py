#22/23 question 2(b)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE dy/dt = 10y^2 - y^3
def model(y, t):
    dydt = 10 * y**2 - y**3
    return dydt

# Function to find ignition delay
def find_ignition_delay(y0):
    # Time points
    t = np.linspace(0, 20, 1000)

    # Solve the ODE
    y = odeint(model, y0, t)

    # Find the index where y crosses a certain threshold (e.g., 0.5 times the initial value)
    threshold = 0.5 * y0
    ignition_index = np.argmax(y > threshold)

    # Ignition delay is the corresponding time
    ignition_delay = t[ignition_index]

    return ignition_delay

# Initial conditions
y0_values = [0.02, 0.01, 0.005]

# Find and report ignition delays
for y0 in y0_values:
    ignition_delay = find_ignition_delay(y0)
    print(f'Ignition delay for y(0) = {y0}: {ignition_delay:.4f} seconds')

# Optional: Plot the solutions
t = np.linspace(0, 20, 1000)
for y0 in y0_values:
    y = odeint(model, y0, t)
    plt.plot(t, y, label=f'y(0) = {y0}')

plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
