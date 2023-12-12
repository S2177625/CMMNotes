import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def gradient_descent(func, gradient_funcs, initial_point, learning_rate, max_iterations, tolerance, minimize=True, color = "red"):
    """
    This algorithim is poor when x approaches a limit, not infinity, eg. log()
    
    It does best with polynomials, specifically quadratics as the only have
    one minima/maxima and it cannot converge upon the wrong turnign point 
    """
    
    
    # changes the direction that the point converges upon
    multplier = 1
    if minimize:
        multplier = -1
        
    new_point = np.array(initial_point)

    for iteration in range(max_iterations):
        old_point = np.array(new_point)
        
        # find the gradient vector
        gradients = np.array([grad_func(new_point[0],new_point[1]) for grad_func in gradient_funcs])

        # step the function in that direction
        for i in range(len(gradients)):
            new_point[i] += multplier*(learning_rate * gradients[i])
            plt.scatter(new_point[0], new_point[1], color=color, marker='o')

        # only if its not the first itteration such that there is an old point
        if max(abs(new_point-old_point)) <= tolerance:
            print(f"Converged in {iteration} itterations")
            return new_point, func(new_point[0], new_point[1])
        # DELETE THIS IF IN USE    
    
    print("possible failure")
    return new_point, func(new_point[0], new_point[1])

N = 33
SIZE = 2
x_vals = np.linspace(-SIZE, SIZE, N)    # x axis values
y_vals = np.linspace(-SIZE, SIZE, N)    # y axis values
X, Y = np.meshgrid(x_vals, y_vals)      # create 2d array from 1d lists


x, y = sym.symbols("x y")
F_sym = 2*x*y + 2*x - x**2 - 2*y**2

dFx_sym = F_sym.diff(x)
dFy_sym = F_sym.diff(y)
dFx = sym.lambdify((x, y), dFx_sym, 'numpy')
dFy = sym.lambdify((x, y), dFy_sym, 'numpy')

F = sym.lambdify((x, y), F_sym, 'numpy')


# Specify initial point and other parameters
initial_point = np.array([0.0, 0.0])
learning_rate = 0.3
max_iterations = 100
tolerance = 0.001

# Call gradient descent for minimization
minima_point, minima = gradient_descent(F,
                                        [dFx, dFy],
                                        initial_point,
                                        learning_rate,
                                        max_iterations,
                                        tolerance,
                                        color = "cyan",
                                        minimize=True)

maxima_point, maxima = gradient_descent(F,
                                        [dFx, dFy],
                                        initial_point,
                                        learning_rate,
                                        max_iterations,
                                        tolerance,
                                        minimize=False,
                                        color = "yellow")

x_max, y_max = maxima_point
x_min, y_min = minima_point

#print(maxima_point, maxima)
plt.scatter(x_max, y_max, color='red', marker='o')
plt.scatter(x_min, y_min, color='blue', marker='o')
plt.scatter(initial_point[0], initial_point[1], color='green', marker='o')

plt.imshow(F(X,Y), cmap='viridis', extent=(x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]), interpolation='nearest', origin='lower')

plt.show()