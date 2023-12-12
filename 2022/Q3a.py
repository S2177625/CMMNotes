import numpy as np

def f(x):
    return 1/np.sin(x) + 1/4

x = 2

#Keep iterating until x_n is very close to x_n+1
while abs(x - f(x)) > 1E-15:
    x = f(x)
    
print(x)