import numpy as np
import scipy
import matplotlib.pyplot as plt

def RMSE(true_x, true_y, x, y):
    print(len(true_x))
    s = 0
    for i in range(len(true_x)):
        s += (y[i]-true_y[i])**2
    rmse = ((1/len(true_x))*(s))**(0.5) 
    return rmse

# define the model of the slope

def model(x,y):
    return y*(1-y)


# set the initial Conditions
x0 = 0
y0 = (np.e**-4)/(np.e**-4+1)

# what region to solve for
x_start = 0
x_end = 10


# Analytical Solution ----------------------------------------------------------
x_exact = np.arange(x_start, x_end, h)
y_exact = (np.e**(x_exact-4))/(np.e**(x_exact-4)+1)



# Numerical Solution -----------------------------------------------------------
x_solveIVP = np.arange(x_start, x_end, h)
solution = scipy.integrate.solve_ivp(model, [x_start, x_end], [y0, x0], t_eval=x_exact)
y_solveIVP = solution.y[0]



# Euler method -----------------------------------------------------------------
x_euler = np.arange(x_start, x_end, h)  # set up x
y_euler = np.linspace(0,0,round((x_end-x_start)/h)) # set up solution

x_euler[0] = x0 # set initial conditions
y_euler[0] = y0 # set initial conditions

for i in range(len(x_euler)-1):  # Apply Euler method n_step times 
    slope = model(x_euler[i], y_euler[i]) # compute the slope using the differential equation
    y_euler[i+1] = y_euler[i] + h * slope  # use the Euler method


# errors
rmse_euler      = RMSE(x_exact, y_exact, x_euler, y_euler)
rmse_solveIVP   = RMSE(x_exact, y_exact, x_solveIVP, y_solveIVP)

errors_ivp.append(rmse_solveIVP)
errors_eul.append(rmse_euler)

print(f"errors : {rmse_euler}, {rmse_solveIVP}")

# Plotting

plt.plot(x_exact, y_exact, c="black", label = "analytical")  # plotting the analytical solution
plt.scatter(x_euler, y_euler, c="blue", label = "euler") # plotting the numerical solution
plt.scatter(x_solveIVP, y_solveIVP, c="red", label = "solve_ivp()") # plotting the numerical solution

plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Solution of the ODE')

plt.show()

