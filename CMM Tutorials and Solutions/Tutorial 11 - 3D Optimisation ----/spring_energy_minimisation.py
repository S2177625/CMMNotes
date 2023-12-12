import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

# CREATE COST FUNCTION ----------------------------------------
x1, x2 = sym.symbols("x1 x2")

ka = 9;   kb = 2
La = 10;  Lb = 10
F1 = 2;   F2 = 4

# Use the given equation for energy and optimise
energy_sym = 0.5*(ka*((x1**2+(La-x2)**2)**0.5 - La)**2)+0.5*(kb*((x1**2+(Lb+x2)**2)**0.5 - Lb)**2)-F1*x1-F2*x2
energy_func = sym.lambdify((x1, x2), energy_sym, 'numpy')

# Energy function
def objective(X):
    x1, x2 = X
    return energy_func(x1, x2)

solution = scipy.optimize.minimize(objective, [0.1, 0.1])
opt_x1, opt_x2 = solution["x"]

print(f"""
D :  {opt_x1:.2f} cm
L :  {opt_x2:.2f} cm
E :  {abs(energy_func(opt_x1, opt_x2)):.2f} J
""")



# DISPLAY ONLY
N = 33
SIZE = 2
length_vals = np.linspace(-5, 5, N)    # x axis values
diameter_vals = np.linspace(-5, 5, N)    # y axis values
X1, X2 = np.meshgrid(length_vals, diameter_vals)      # create 2d array from 1d lists

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, energy_func(X1, X2), cmap=cm.magma)

plt.show()