import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#constants
m = 7850 #kg/m^3
E = 200 * 10**9 #Gpa to Pa
L = 0.9 #m
I = 3.255 * 10**-11 #m^4

#grpahing the zeroes
yB = lambda B: np.cosh(B)*np.cos(B) + 1
b=np.arange(-2,2,0.1)
plt.plot(b, yB(b))
plt.show()

#numerically solving
initial_guess = -1, 1
natural_frequencies = fsolve(yB, initial_guess)
print(natural_frequencies)

f = lambda fi: ((2*np.pi*fi)**2)*(m*L**3)/(E*I)-natural_frequencies**4
frequencies = fsolve(f, initial_guess)

print(frequencies)

fi = np.sqrt((natural_frequencies**4)/((m*L**3)/(E*I)))/(2*np.pi)
print(fi)