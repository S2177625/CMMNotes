import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

L_val = 800
E_val = 40000
I_val = 40000
w_val = 3.5

n_step = 100

L = 1000

y = np.linspace(0,L,n_step)
x = np.linspace(0,L,n_step)

def model(x):
    yy = (w / (120 * E * I * L))*-x**5 + (w / (120 * E * I * L))*2*L**2*x**3 - (w / (120 * E * I * L))*L**4*x
    return yy


for i in range(len(x)):
    E = 40000
    L = 800
    I = 40000
    w = 3.5
    y[i] = model(x[i])
    
plt.plot(x,y)
plt.show()


scipy.optimize.root(model, [1000])

  
  
    
        




    
