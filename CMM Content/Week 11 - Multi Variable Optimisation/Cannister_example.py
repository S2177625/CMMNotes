import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

# STATIC VALUES
volume = 0.8
thickness   = 0.03
density     = 8000

max_diameter = 1
max_length   = 2

mass_cost_per_kg    = 4.5
weld_cost_per_meter = 20

length_initial   = max_diameter
diameter_initial = max_length



# CREATE COST FUNCTION ----------------------------------------
d, l = sym.symbols("x y")

V_cylinder  = l*np.pi*((d/2+thickness)**2-(d/2)**2)
V_plate     = np.pi*((d/2+thickness)**2)*thickness
V_total     = 2*V_plate +  V_cylinder
mass_total  = V_total * density
weld_tot = 4*np.pi*(d+thickness)
material_cost = mass_total*mass_cost_per_kg
weld_cost = weld_tot*weld_cost_per_meter

cost_sym = material_cost + weld_cost
cost_func = sym.lambdify((d, l), cost_sym, 'numpy')


# CREATE CONSTRAINT FUNCTION  -----------------------------------
# function for constraint AND subtract the value you want
constraint_sym = l*np.pi*(d/2)**2 - volume
constraint_func = sym.lambdify((d, l), constraint_sym, 'numpy')

# cost function
def objective(X):
    diameter, length = X
    return cost_func(diameter,length)

# constraint function, where this equals zero
def eq(X):
    diameter, length = X
    return constraint_func(diameter,length)

bounds = [(0, max_diameter), (0, max_length)]
solution = scipy.optimize.minimize(objective, [0.5, 0.5], constraints={'type': 'eq', 'fun': eq}, bounds=bounds)

opt_diameter, opt_length= solution["x"]
print(f"""
D    :  {opt_diameter*100:.2f} cm
L    :  {opt_length*100:.2f} cm
Cost : £{cost_func(opt_diameter, opt_length):.2f}
""")



# DISPLAY ONLY
N = 33
SIZE = 2
length_vals = np.linspace(0, max_length, N)    # x axis values
diameter_vals = np.linspace(0, max_diameter, N)    # y axis values
L, D = np.meshgrid(length_vals, diameter_vals)      # create 2d array from 1d lists

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(D, L, cost_func(D,L), cmap=cm.magma)

plt.show()