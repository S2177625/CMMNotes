import matplotlib.pyplot as plt
import numpy as np

#Variables
L = 800 #cm
E = 40000 #kN/cm^2
I = 40000 #cm^4
w0 = 3.5 #kN/cm
x = np.linspace(0,L,50)

def model(x):
    temp = (w0/(120*E*I*L))*((-x**5)+((2*L**2)*(x**3))-(L**4*x))
    return temp

y = model(x)

plt.plot(x,y)
plt.show()
