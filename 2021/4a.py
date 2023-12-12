#2021 2022 question 4a

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(t,y):
    f_1 = -10*y + 11*np.cos(t) - (-9)*np.sin(t)
    return [f_1]
# ------------------------------------------------------


# ------------------------------------------------------
# initial conditions
t0 = 0
y0 = 1
# total solution interval
t_final = 4*np.pi
# step size
h = 0.1
# ------------------------------------------------------


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
    [slope_1] = model(t_eul[i],y_eul[i]) 
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope_1
    print(y_eul[i])
# ------------------------------------------------------

#solutions at t=2pi and t= 4pi
y_2pi = y_eul[int(2*np.pi / h)]
y_4pi = y_eul[-1]

print("the solution of y at t=2*np.pi is",  y_2pi)
print("the solution of y at t=4*np.pi is",  y_4pi)


# ------------------------------------------------------
# plot results
plt.plot(t_eul, y_eul , 'b.-')
plt.xlabel('t')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------

# ------------------------------------------------------
# print results in a text file (for later use if needed)
file_name= 'output_h' + str(h) + '.dat' 
f_io = open(file_name,'w') 
for i in range(n_step+1):
    s1 = str(i)
    s2 = str(t_eul[i])
    s3 = str(y_eul[i])
    s_tot = s1 + ' ' + s2 + ' ' + s3
    f_io.write(s_tot + '\n')
f_io.close()
# ------------------------------------------------------

