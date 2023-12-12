#Question 2

#Part A

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x):
    
    dydt = 10*y**2-y**3
    
    return dydt

# initial conditions
t0 = 0
y0 = 0.02
# total solution interval
t_final = 10
# step size
h = 0.01
# -----------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(t_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
t_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
t_eul[0] = t0 

# Populate the x array
for i in range(n_step):
    t_eul[i+1]  = t_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],t_eul[i]) 
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution 
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Definition of array to store the exact solution
n_exact = 1000
t_exact = np.linspace(0,t_final,n_exact+1) 
y_exact = np.zeros(n_exact+1)

# exact values of the solution
for i in range(n_exact+1):
    y_exact[i] = y0 * math.exp(-t_exact[i])
# ------------------------------------------------------

# ------------------------------------------------------
# print results on screen
print ('Solution: step t y-eul y-exact error%')
for i in range(n_step+1):
    print(i,t_eul[i],y_eul[i], y0 * math.exp(-t_eul[i]),
            (y_eul[i]- y0 * math.exp(-t_eul[i]))/ 
            (y0 * math.exp(-t_eul[i])) * 100)
# ------------------------------------------------------

# ------------------------------------------------------
# print results in a text file (for later use if needed)
file_name= 'output_h' + str(h) + '.dat' 
f_io = open(file_name,'w') 
for i in range(n_step+1):
    s1 = str(i)
    s2 = str(t_eul[i])
    s3 = str(y_eul[i])
    s4 = s1 + ' ' + s2 + ' ' + s3
    f_io.write(s4 + '\n')
f_io.close()
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.plot(t_eul, y_eul , 'b.-',t_exact, y_exact , 'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------

# Results: at t=4 y= 0.086, t=6 y=1.04, t=10 y= 10
#The system crashes/unstable at around t=9 as there is a sudden change in the graphs behavoiu
#Thus to solve the issue a very small step size of 0.001 was chosen creating 10000 iterations 



# Part B

#For part B the intial conditions are changed to test for the value of the ignition delay

# approximatley from A at y(0)=0.02 the ignition lag is t= 4.8, y(0)=0.01 t=9.8, y(0)=0.005 t= 1.9



# Part C

#Testing the threshold step before system unstability:
    
# It is found that a tthe given initial conditions the system becomes unstable at h= 0.01
#Any larger time step and the system will start displaying unstability


#D 

#Answer can be found in the slides (Week 7 lecture)
#The Euler method becomes unstable as the step size is increased because larger steps can 
#overshoot the true solution, amplifying any errors and potentially leading to divergent results, especially for stiff differential equations or when the problem's domain has rapid changes.

