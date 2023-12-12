# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:57:03 2020

@author: emc1977
"""

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline



def trapz(f,a,b,N=1000):
    '''Approximate the integral of f(x) from a to b by the trapezoid rule.

    The trapezoid rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/2) \sum_{k=1}^N (f(x_k) + f(x_{k-1}))
    where x_k = a + k*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using the
        trapezoid rule with N subintervals of equal length.

    Examples
    --------
    >>> trapz(np.sin,0,np.pi/2,1000)
    0.9999997943832332
    '''
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N
    total = (dx/2) * np.sum(y_right + y_left)
    return total

func = lambda x: np.exp(-x**2)
#def func(x):
#    return np.exp(-x**2)

x = np.linspace(-2,2,100)
y = func(x)


# range to integrate
a = -1.8
b = 1.8

plt.plot(x,y)
area = trapz(func,a,b,N=7)

plt.show()
