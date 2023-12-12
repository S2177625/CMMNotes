##QUESTION 3 iterative equation root finding


def newton(f,Df,x0,epsilon,max_iter):
    
    
##Open root finding method
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

f = lambda x: 1/np.sin(x) + 1/4 - x
df= lambda x: -np.cos(x)/(np.sin(x))**2 - 1
x0=1
epsilon=0.00001
max_iter=100
solution = newton(f,df,x0,epsilon,max_iter)
print(f'solution: {solution}')






##closed root finding method

import math
import numpy as np
##After initially plotting function, suitable bounds were chosen for the bisection method

def bisection(f, a, b, N):
    '''Approximate solution of f(x) = 0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x) = 0.
    a, b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a) * f(b) >= 0 since a solution is not guaranteed.
    N : positive integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and returns None.
    '''
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None

    a_n = a
    b_n = b

    for n in range(1, N + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)

        if f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None

    return (a_n + b_n) / 2

# Example usage with your function
f = lambda x: 1/np.sin(x) + 1/4 - x
approx_phi = bisection(f, 1, 1.5, 3)
print(f"answer is {approx_phi}")


