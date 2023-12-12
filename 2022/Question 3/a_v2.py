# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:29:33 2020

@author: emc1977
"""
# Fixed Point Iteration Method
# Importing math to use sqrt function
import math
import numpy as np

# Input Section
x0 = 0
e = 1e-3
N = 1000

# Converting x0 and e to float
x0 = float(x0)
e = float(e)

# Fixed Point Iteration Method
# Importing math to use sqrt function
import math

def f(x):
    return 1/(np.sin(x)) + 1/4 - x

# Re-writing f(x)=0 to x = g(x)
def g(x):
    return 1/(np.sin(x)) + 1/4
    
# Implementing Fixed Point Iteration Method
def fixedPointIteration(x0, e, N):
    print('\n\n*** FIXED POINT ITERATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        x1 = g(x0)
        print('Iteration-%d, x1 = %0.6f and f(x1) = %0.6f' % (step, x1, f(x1)))
        x0 = x1

        step = step + 1
        
        if step > N:
            flag=0
            break
        
        condition = abs(f(x1)) > e

    if flag==1:
        return x1
        # print('\nRequired root is: %0.8f' % x1)
    else:
        print('\nNot Convergent.')

# Input Section
e = 0.001
N = 100

# Converting x0 and e to float
x0 = float(x0)
e = float(e)

# Converting N to integer
N = int(N)


#Note: You can combine above three section like this
# x0 = float(input('Enter Guess: '))
# e = float(input('Tolerable Error: '))
# N = int(input('Maximum Step: '))

# Starting Newton Raphson Method
solutions = []
mylist = np.linspace(-1000, 1000, 1000)

for j in range(len(mylist)):
    x0 = mylist[j]
    myappend = True
    current_sol = fixedPointIteration(x0,e,N)
    for i in range(len(solutions)):
        if current_sol < (solutions[i] + 2*e) and current_sol > (solutions[i] - 2*e):
            myappend = False
    if current_sol < 0 or current_sol > 4:
        myappend = False
    if myappend == True:
         solutions.append(current_sol)

print(solutions)

"""
NOTES

Tolerable error
Because you have a tolerance, you will never return to the same x

A loop that tries different guesses and then outputs all the solutions and then 
You start from an array of guesses (e.g. euqally spaced between -10 and 10).
Yourun your method. You get your solution
compare your soolution with the previous solutions
If the new solution is within the method tolerance of a previous solution, 
discard the new one.
Otherwise you save it. 

there is a massiv edifference between finding one root and then finding all the roots. 
Finding all the roots is a massive endeavour. 
Cooking rice analogy.

In order to compare the absolute errors, you should comapre them using the same tolerance.
Errors and stuff: 

Closed interval methods are more difficult to guide towards the solution. 

"""