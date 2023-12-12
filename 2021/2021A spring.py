import numpy as np
import matplotlib.pyplot as plt

A_j = 0.1 #m
phi = np.pi/8

c_over_m = 12 #s^-1
s_over_m = 1500 #s^-2

coefficients = [1, 2*c_over_m, 3*s_over_m, c_over_m*s_over_m, s_over_m**2]

roots = np.roots(coefficients)

print(roots)

wj1 = roots[0]
wr1 = wj1.real
wi1 = wj1.imag

wj2 = roots[1]
wr2 = wj1.real
wi2 = wj2.imag

t=np.arange(0,11, 0.1)
def calculate_displacement(wr, wi, t):
        xj = A_j*np.e**(wr*t)*np.cos(wi*t + phi)
        return xj

x2 = calculate_displacement(wr2, wi2, t)

plt.plot(t, x2)
plt.title('time vs displacmeent')
plt.xlabel('time')
plt.ylabel(('displacement'))
plt.show()

F = 100 #N
displacement = x2[-1] - x2[0]
work = abs(F * displacement)
print(f'minimum work done by x2 in 10 seconds is {work:.2f} Joules')