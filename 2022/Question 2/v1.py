# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x): # x is t!
    # k= 1
    dydx = 10*y**2 - y**3
    return dydx

# initial conditions
x0 = 0
y0 = 0.02 # y(0) = 0.02
# total solution interval
x_final = 25  # Don't try to change the typ eof a variable (single int)
# step size
h = 0.02
# I foyu want to have a creiterion to identify auto, after the ignition delay, the sol becomes flat. 
# The best way is that when you start to see oscillations, thenyou have a problem. 
# Threshold is around 0.02 (oscillation)
# WHEN THE H IS TOO BIG, THE THING CAN EXPLODE. 
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1, dtype = float)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
x_eul[0] = x0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i]  + h

slope_array = []
slope_array = np.zeros(n_step+1)

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],x_eul[i]) 
    # use the Euler method
    slope_array[i] = slope
    y_eul[i+1] = y_eul[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution 
# n_exact linearly spaced numbers
# only needed for plotting reference solution

"""
# Definition of array to store the exact solution
n_exact = 1000
x_exact = np.linspace(0,x_final,n_exact+1) 
y_exact = np.zeros(n_exact+1)

# exact values of the solution
for i in range(n_exact+1):
    y_exact[i] = y0 * math.exp(-x_exact[i])

# ------------------------------------------------------

# ------------------------------------------------------
# print results on screen
print ('Solution: step x y-eul y-exact error%')
for i in range(n_step+1):
    print(i,x_eul[i],y_eul[i], y0 * math.exp(-x_eul[i]),
            (y_eul[i]- y0 * math.exp(-x_eul[i]))/ 
            (y0 * math.exp(-x_eul[i])) * 100)
# ------------------------------------------------------

# ------------------------------------------------------
# print results in a text file (for later use if needed)
file_name= 'output_h' + str(h) + '.dat' 
f_io = open(file_name,'w') 
for i in range(n_step+1):
    s1 = str(i)
    s2 = str(x_eul[i])
    s3 = str(y_eul[i])
    s4 = s1 + ' ' + s2 + ' ' + s3
    f_io.write(s4 + '\n')
f_io.close()
# ------------------------------------------------------
"""

# Every time you make a for loop, make it clear which is the 
# initial and which is the final step and why.

epsilon = 0.001
t_sol1 = 4
t_sol2 = 5
for i in range(n_step + 1):
    if x_eul[i] <= (t_sol1 + epsilon) and x_eul[i] >= (t_sol1 - epsilon):
        print("soltion at t = 4:" + str(y_eul[i]))
    if x_eul[i] <= (t_sol2 + epsilon) and x_eul[i] >= (t_sol2 - epsilon):
        print("soltion at t = 5:" + str(y_eul[i]))
    if slope_array[i] <= (max(slope_array) + epsilon) and slope_array[i] >= (max(slope_array) - epsilon):
        print(x_eul[i])


# print(max(slope_array))



# ------------------------------------------------------
# plot results
plt.plot(x_eul, y_eul , 'b.-')#,x_exact, y_exact , 'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------




"""
NOTES

Easy an not robust, and then not so easy but more robust. 

I have the array with the solutions. I can find the index where the time = 4 and then go back to my 
y solution and then find the value of that particular index

Find mroe or less waht you expect to see. 
For the first one, around 5 seconds. for the second, 10, for the last, 20. 


If the value of the slope (iterative) is larger than the slope at t = 0 by x amount, stop
"""