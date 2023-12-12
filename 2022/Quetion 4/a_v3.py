import scipy
import numpy as np
import matplotlib.pyplot as plt

def myequation(x, t):
    m = 1
    k = 1
    b = 1
    return [m*x[1], -k*x[0] -b*x[1]]

trange = np.arange(0, 10, 0.1)

solution = scipy.integrate.odeint(myequation, [0, 0.05], trange)

print(solution)

plt.plot()

"""
ODEint can 

You cannot use chatGPT. But it does nto say that you cannot use something for cahtgpt. 

"""