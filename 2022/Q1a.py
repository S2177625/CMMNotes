import numpy as np

#Using a function we can change the number of iterations
def pi(N):
    temp = 0
    for i in range(1,N+1):
        temp += 1/(i**2)
    pi = np.sqrt(6*temp)
    return pi

print('At N=10, pi=',pi(10))
print('At N=100, pi=',pi(100))
print('At N=1000, pi=',pi(1000))