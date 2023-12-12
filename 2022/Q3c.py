import numpy as np

#Equation function
def f(x):
    return 1/np.sin(x) + 1/4 - x

#Differential function
def df(x):
    return -1/(np.tan(x)*np.sin(x)) - 1

#Number of iterations and initial value
N = 30000
x = 2

#Newton Raphson
for i in range(N):
    x = x - f(x)/df(x)

print(x)