import numpy as np
import matplotlib.pyplot as plt
import math


def model(t, y):
    parameter = -10
    dydt = parameter*y + (1-parameter)*np.cos(t) - (1+parameter)*np.sin(t)
    return dydt

def euler_method(t_final, step_sizes):
    '''
    There are two different step sizes that can be switched between at different ranges
    in case the ODE is stiff. Through trial and error it is evident that the ODE
    is not difficult, and a step size of 0.01 is quick to execute, and provides a smooth
    line on the graph
    '''
    #initial conditions
    t0 = 0
    y0 = 1
    
    # initial and secondary step sizes, equivalent for this scenario
    if isinstance(step_sizes, tuple):
        h1, h2 = step_sizes
    else:
        h1 = h2 = step_sizes

    #change step size at t=t2
    t2 = t_final/2
    
    # round up number of steps
    n_steps = math.ceil(t2/h1) + math.ceil((t_final-t2)/h2)
    
    # number of elements in the array should be equal
    t_eul, y_eul = np.zeros(n_steps+1), np.zeros(n_steps+1)
    
     #initialize initial conditions
    t_eul[0], y_eul[0] = t0, y0

    # Populate time array accounting for step size
    for n in range(n_steps):
         # interval of h1 if t<t2, interval of h2 if t>t2
        if t_eul[n]<t2:
            t_eul[n+1] = t_eul[n] + h1
        else:
            t_eul[n+1] = t_eul[n] + h2
            
    for n in range(n_steps):
       # slope = dy/dt
        slope = model(t_eul[n], y_eul[n]) 
        if t_eul[n]>t2:
            h = h2
        else: 
            h = h1
        #add stepsize * slope to previous value
        y_eul[n+1] = y_eul[n] + h*slope 
    '''
    print(f'for time {t_eul[-1]:.2f} final solution is {y_eul[-1]:.2f}')
    plt.plot(t_eul, y_eul)
    plt.xlabel('time')
    plt.ylabel('response')
    plt.show()
    '''
    
    def equation(t):
        y = np.sin(t) + np.cos(t)
        return y

    def calculate_max_true_err(t, num_vals):
        true_vals = equation(t)  
        abs_err = true_vals-num_vals #this is an array    
        return np.max(abs_err)
        
    max_err = calculate_max_true_err(t_eul, y_eul)

    return t_eul, y_eul, max_err #these are arrays

#solution intervals
t_final1 = 2*np.pi
t_final2 = 4*np.pi

#step_sizes = 0.01, 0.001
step_size_trials = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

max_errors1, max_errors2 = [], []

for item in step_size_trials: 
    time1, solution1, max_err1 = euler_method(t_final1, item)
    time2, solution2, max_err2 = euler_method(t_final2, item)
    
    max_errors1.append(max_err1)
    max_errors2.append(max_err2)

print(step_size_trials)
print(max_errors1)
print(max_errors2)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(step_size_trials, max_errors1)
ax1.set_title('Maximum error over a period of 2 pi')
ax2.plot(step_size_trials, max_errors2)
ax2.set_title('Maximum error over a period of 4 pi')
plt.show()


'''
It is clear that as step size increases, the error increases linearly or slightly exponentially
step sizes of 0.25 and 0.3 are unstable creating errors that are incredibly high

'''