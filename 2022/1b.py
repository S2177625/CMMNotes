import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation
def f(t, y):
    return 10 * y**2 - y**3

# Euler's method
def euler_method(f, y0, t0, tn, h):
    t_values = np.arange(t0, tn + h, h)
    y_values = [y0]

    for t in t_values[:-1]:
        y0 = y0 + h * f(t, y0)
        y_values.append(y0)

    return t_values, y_values

# Function to find ignition delay
def find_ignition_delay(t_values, y_values, threshold):
    for t, y in zip(t_values, y_values):
        if y > threshold:
            return t

    return None

# Initial conditions
initial_conditions = [0.02, 0.01, 0.005]

# Parameters
t0 = 0
tn = 5  # Adjust as needed
h = 0.01

# Plotting
plt.figure(figsize=(10, 6))

for y0 in initial_conditions:
    t_values, y_values = euler_method(f, y0, t0, tn, h)
    plt.plot(t_values, y_values, label=f'y(0) = {y0}')

    # Find ignition delay
    ignition_delay = find_ignition_delay(t_values, y_values, threshold=0.1)
    print(f'Ignition delay for y(0) = {y0}: {ignition_delay:.4f} units of time')

plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.title('Numerical Solution using Euler\'s Method')
plt.show()
