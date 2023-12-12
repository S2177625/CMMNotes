import numpy as np
import matplotlib.pyplot as plt

x1_a = np.linspace(-10,10,100)
x2_a = np.linspace(-10,10,100)

x1, x2 = np.meshgrid(x1_a, x2_a, indexing='ij')

ka=9.0
kb=2.0
La=10.0
Lb=10.0
F1=2.0
F2=4.0

PE = 0.5*(ka*((x1**2+(La-x2)**2)**0.5 - La)**2)+0.5*\
    (kb*((x1**2+(Lb+x2)**2)**0.5 - Lb)**2)-F1*x1-F2*x2


fig = plt.figure(figsize = (8,6))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x1, x2, PE, cmap = plt.cm.viridis)

plt.xlabel('x_1')
plt.ylabel('x_2')

plt.show()
