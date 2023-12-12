
##QUESTION 1: truncate approximation
import numpy as np

def truncate(N):
    sqrd = 0
    for i in range(1,N +1):
        sqrd_old = sqrd
        sqrd += 6 * 1/i**2
        
    approx = np.sqrt(sqrd)
    true_error = np.pi - approx
    Rel_error = (np.sqrt(approx - np.sqrt(sqrd_old)))/approx
    print(approx)
    print(f'true error is {true_error}')
    print(f'Relative error is {Rel_error}')
    print("")
    
truncate(10)
truncate(100)
truncate(1000)

##EXAMple solution for relative error
##Relative error defined as absolute value of (final approx - previous approx)
import numpy as np
import matplotlib.pyplot as plt
import math
N = 100
s=0
pi_n = np.zeros(N)
nn = np.zeros(N)
error_true = np.zeros(N)
error_ext = np.zeros(N)
for i in range(1,N+1):
    pi_old = (s*6.0)**0.5
    s = s + 1.0/i**2.0
    pi_n[i-1] = (s*6.0)**0.5
    nn[i-1] = i
error_true[i-1] = np.absolute(pi_n[i-1] - np.pi)
error_ext[i-1] = np.absolute(pi_n[i-1] - pi_old)
print(i,pi_n[i-1],error_true[i-1], error_ext[i-1])
