import numpy as np
import sympy as sym

#PART A

roots = np.roots([1, 24, 4500, 12*1500, 1500**2])
wr = roots.real
wi = roots.imag
print('First pair: wr =', wr[0], 'wi =', wi[0])
print('Second pair: wr =', wr[2], 'wi =', wi[2])

#PART B

#Work done is the integral of force with respect to its displacement
t, k, F, W = sym.symbols('t k F W')

A = 0.1
wr = wr[0]
wi = wi[0]
phi = sym.pi/8

k = 100/A #for some reason

x = A*sym.exp(wr*t)*sym.cos(wi*t + phi) #Displacement function
F = -k*x #Force function
W = sym.integrate(F, (t, 0, 10)).evalf()
print('Minimum work is', W)