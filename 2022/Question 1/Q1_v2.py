# Don't jsut jump into it. 
# Two things: one is compute the value, and another is comptur the error. 
# We need to take a number and the number is given. 

# Return pi approximation, the true error, and the estimated error. 
# Start off with your returns. You can always remove them.
# Every time you introduce a funciton, it has to be very clear what it is. 


import math
from math import pi
import numpy as np

def equation(N):
    sum = 0
    for i in range(1, N+1):
        sum += 1/(i**2)
    pi_approx = math.sqrt(6*sum) # variable sum is the summation of 1/i^2 terms.
    1/1+1/4+1/9

    true_error = ((pi-pi_approx)/pi) * 100
    return pi_approx , true_error #, estimated_error

k = 3
N_array = np.logspace(1, k, k)
pi_result = []
true_error = []
approx_error = []
# Be careful! Will approx_error have the same space of all the other values?
pi_result = [0 for i in range(k)]
true_error = [0 for i in range(k)]
approx_error = [0 for i in range(k)]

for i in range(0, k):
    pi_result[i], true_error[i] = equation(int(N_array[i]))
    if i > 0:
        approx_error[i-1] = ((pi_result[i] - (pi_result[i-1]))/pi_result[i])*100

print(pi_result, true_error, approx_error)

# Now we want to compute the approximate error. 
# Now I need to access the i-1th iteration and do (i - (i-1))/i
# We want i to be the approximation of i corresponding to this iteration.
# If I am outside of the loop, i will take the last value that it was in the loop.

# Try using only those three values and figure out how to best guess the value of pi based on those three 
# Hint: difference between first and second is 0.1, the erorr between first and second is 
# Function to find the error. 

"""
NOTES 

Initialising variables is important for computational cost. 
I you want to do it over many many iterations, 
If you know waht size the array is taht you want it to be, then initialise it.
It's good habit to intialise your variables. This is how you write good code that is robust.

Always put a string to something as late as possible. 
Always keep something as a number as late as you can.

You want functions as specific possible. To an extent, small. 
The important thing is that this funciton works. 
We know it works. We know how it works. 

Convergence.
When you have a numerical method and oyu don't know the answer, 

Tolerance is when you have no specific knowledge of the true value. If the difference between two 
consecutive solutions is less than whatever I want, then it's 
There are methods that oscillate etc. 
For some methods, you don't know your exact 

print(equation(10))
print(equation(100))
print(equation(1000))
"""