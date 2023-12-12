# Python3 implementation of Bisection 
# Method for solving equations

MAX_ITER = 1000000

# An example function whose solution 
# is determined using Bisection Method.  
# The function is x^3 - x^2 + 2 
def func(x): 
    return (x**2 + 5*x - 4)

# Prints root of func(x) in interval [a, b] 
def regulaFalsi(a, b): 
    if func(a) * func(b) >= 0: 
        print("You have not assumed right a and b") 
        return -1

    c = a  # Initialize result 

    for i in range(MAX_ITER): 
        # Find the point that touches x axis 
        c = (a * 0.5 * func(b) - b * func(a)) / (0.5 * func(b) - func(a))

        # Check if the above-found point is the root 
        if func(c) == 0: 
            break

        # Decide the side to repeat the steps 
        elif func(c) * func(a) < 0: 
            b = c 
        else: 
            a = c 
    
    true_value = 2  # Assuming 2 as the true root for the example function
    true_error = abs((true_value - c) / true_value) * 100
    
    print("The value of root is:", '%.4f' % c)
    print("True Error:", '%.4f' % true_error, "%")
    print("Number of iterations:", i)

# Driver code to test above function 
# Initial values assumed 
a = -5
b = 1
regulaFalsi(a, b)
