# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:13:01 2020

@author: A. Fragkou & E. McCarthy
"""

import numpy as np

def linearsolver(A,b):
    n = len(A)

    #Initialise solution vector as an empty array
    x = np.zeros(n)

    #Join A and use concatenate to form an augmented coefficient matrix
    M = np.concatenate((A,b.T), axis=1)

    for k in range(n):
        for i in range(k,n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[[k,i]] = M[[i,k]]
            else:
                pass
                for j in range(k+1,n):
                    q = M[j][k] / M[k][k]
                    for m in range(n+1):
                        M[j][m] +=  -q * M[k][m]

    #Python starts indexing with 0, so the last element is n-1
    x[n-1] =M[n-1][n]/M[n-1][n-1]

    #We need to start at n-2, because of Python indexing
    for i in range (n-2,-1,-1):
        z = M[i][n]
        for j in range(i+1,n):
            z = z  - M[i][j]*x[j]
        x[i] = z/M[i][i]

    return x

#Initialise the matrices to be solved.
#A=np.array([[10., 15., 25],[4., 5., 6], [25, 3, 8]])
#b=np.array([[34., 25., 15]])
#print(linearsolver(A,b))

m1, m2, m3 = 70, 60, 40 

c1, c2, c3 = 10, 14, 17

g = 9.81
v = 5

# A is the matric of the coefficients of the un knowns
# the rows are horizontal (like normal)
A=np.array([[m1, 1, 0],
            [m2, -1, 1],
            [m3, 0, -1]])

# b is the matrix of the knowns 
b=np.array([[m1*g - c1*v,
             m2*g - c2*v,
             m3*g - c3*v]])

solution = linearsolver(A,b)

print(f"acceleration : {solution[0]:.4g} m*s^-2")
print(f"Tension in T : {solution[1]:.4g} N")
print(f"Tension in R : {solution[2]:.4g} N")

