import numpy as np
import Q1a

#When the true solution is known, we use the pi value in the numpy library
print('Assuming the true solution is known:')
print('The error for N=10 is',abs(Q1a.pi(10)-np.pi)/np.pi)
print('The error for N=100 is',abs(Q1a.pi(100)-np.pi)/np.pi)
print('The error for N=1000 is',abs(Q1a.pi(1000)-np.pi)/np.pi)

#When the true solution is not known, we use the value of the next higher N
print('\nAssuming the true solution is not known:')
print('The error for N=10 is',abs(Q1a.pi(10)-Q1a.pi(100))/Q1a.pi(100))
print('The error for N=10 is',abs(Q1a.pi(100)-Q1a.pi(1000))/Q1a.pi(1000))
print('The error for N=10 is',abs(Q1a.pi(1000)-Q1a.pi(10000))/Q1a.pi(10000))