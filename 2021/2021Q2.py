import numpy as np
from scipy.optimize import fsolve

# Given parameters
m = 7850  # kg/m^3
L = 0.9  # m
E = 200e9  # Pa (or N/m^2)
I = 3.255e-11  # m^4

# Function defining the frequency equation
def frequency_equation(beta):
    return np.cosh(beta) * np.cos(beta) + 1

# Function to calculate the roots of the frequency equation
def find_roots():
    # Define the equation to solve
    equation_to_solve = lambda beta: frequency_equation(beta)

    # Initial guesses for roots
    initial_guesses = np.linspace(0.1, 20, 10)  # Adjust the range as needed

    # Use fsolve to find roots
    roots = fsolve(equation_to_solve, initial_guesses)

    return roots

# Convert roots to natural frequencies using the relationship: fi = beta_i^2 / (2 * pi)*(np.sqrt(m*L**3/E*I))
def calculate_frequencies(roots):
    frequencies = [beta_i**2 / (2 * np.pi)*(np.sqrt(m*L**3/E*I)) for beta_i in roots]
    return frequencies

# Find roots and corresponding natural frequencies
roots = find_roots()
frequencies = calculate_frequencies(roots)

# Print the results
print("Roots (β values):", roots)
print("Natural Frequencies (fi values):", frequencies)

