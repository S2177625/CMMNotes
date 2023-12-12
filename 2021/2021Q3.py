#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:39:16 2023

@author: ben
"""
import numpy as np
#  part a
'''False position method '''

#  part b
''' corrected syntax errors, changed the names of functions so they can be called'''

MAX_ITER = 1000000

def func(x):
    return (x**3 - 4*(x*2) + 10)

def Code(a,b):
    if func(a) * func(b) >= 0:
        print("You have not assumed correct values of a and b")
        return -1
    c = a 
    
    for i in range(MAX_ITER):
        c = (a * func(b) - b * func(a))/ (func(b) - func(a))
        if func(c) == 0:
            break
        
        elif func(c) * func(a) < 0: 
            b=c
        
        else: 
            a=c
        
    print("The value of root is : " , '%.4f' %c) 
    print(i)
    
#  part c -------------------------------------------------------
    
MAX_ITER = 1000000

def func(x):
    return (x**2 + 5*x -4)

def code(a,b):
    if func(a) * func(b) >= 0:
        print("You have not assumed correct values of a and b")
        return -1
    c = a 
    
    for i in range(MAX_ITER):
        c = (a * func(b) - b * func(a))/ (func(b) - func(a))
        if func(c) == 0:
            break
        
        elif func(c) * func(a) < 0: 
            b=c
        
        else: 
            a=c
        
    print("The value of root is : " , '%.4f' %c) 
    print(i) 
    
a =-3
b = 3
code(a, b) 

#  part d -------------------------------------------------------

''' using symbolab to find true solution'''

MAX_ITER = 1000000

def func(x):
    return (x**2 + 5*x -4)


    

def code(a,b):
    if func(a) * func(b) >= 0:
        print("You have not assumed correct values of a and b")
        return -1
    c = a 
    
    c_true = ((-5+np.sqrt(41))/2)


    for i in range(MAX_ITER):
        c = (a * func(b) - b * func(a))/ (func(b) - func(a))
        absolute_error = abs(c - c_true)
        if func(c) == 0:
            break
        
        elif func(c) * func(a) < 0: 
            b=c
        
        else: 
            a=c
        
    print("The value of root is : " , '%.4f' %c) 
    print(i) 
    print(absolute_error)
    
    
a =-3
b = 3
code(a, b) 

#  part d with aproximate error instead of true -------------------

MAX_ITER = 1000000

def func(x):
    return (x**2 + 5*x -4)


    

def code(a,b):
    if func(a) * func(b) >= 0:
        print("You have not assumed correct values of a and b")
        return -1
    c = a 

    for i in range(MAX_ITER):
        c_prev = c  # Store the previous value of c
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        
        absolute_error = abs(c - c_prev)
        
        if func(c) == 0 or absolute_error < 1e-10:  # Consider a very small absolute error as convergence
            break
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c
        
    print("The value of root is : " , '%.4f' %c) 
    print(i) 
    print(absolute_error)
    
    
a =-3
b = 3
code(a, b) 
