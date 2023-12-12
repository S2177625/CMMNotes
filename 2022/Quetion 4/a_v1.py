import scipy
import numpy as np
import matplotlib.pyplot as plt

def model(y,x):
    m = 1
    b = 1
    k=1
    d2ydx2 = m*y
    dydx = b*y
    x = k*y
    return d2ydx2 + dydx + x

y0 = 0.05

x = np.linspace(0, 10)

y = scipy.integrate.odeint(model, y0, x)

plt.plot(x, y)
plt.show()

"""
ODEINT won't work because second order.

Transformation x2 = y'. 

"""
