import sympy as sp
MAX_ITER = 1000000

#polynomial function
def func(x):
    LHS = 5
    RHS = x**2 + 5*x + 1
    zero = RHS - LHS
    return zero

'''
algorithm being used by the code is the secant method, which aproximates a solution
on interval [a,b] by the secant method.

f : function
    The function for which we are trying to approximate a solution f(x)=0.
a,b : numbers
    The interval in which to search for a solution. The function returns
    None if f(a)*f(b) >= 0 since a solution is not guaranteed.
N : (positive) integer
    The number of iterations to implement.
    
Returns

m_N : number
    The x intercept of the secant line on the the Nth interval
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
        
    The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
    for some intercept m_n then the function returns this solution.
    
    If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
    iterations, the secant method fails and return None.
'''

def code(a,b):
    if func(a) * func(b) >= 0: # y1 * y2 must be opposite signs 
        print("You have not assumed correct values of a and b")
        return None #this line is unecessary
    
    #iterate over a range to check if it's reaching 0
    for i in range(MAX_ITER+1):
        # correct form is c = x1 - (y1*(x2-x1)) / (y2-y1))
        c = a - (func(a)*(b-a) / (func(b)-func(a))) # this is x3, func(c) is y3
        
        if func(c) == 0:
            print ('found exact solution')
        elif func(a) * func(c) < 0: # if y1 * y3 < 0, then they are opposite signs
            b=c #                   # therefore the next guess can use that value of x3 to find y2
        elif func(b) * func(c) < 0:
            a=c
        else:
            print('secant method fails')
    
    x = sp.symbols('x')
    equation = sp.Eq(func(x), 0)
    true_root = sp.solve(equation, x)[0]
    true_value = true_root.evalf()
    print(true_value)
    
    numerical_error = c-true_value
    relative_error = (true_value-c)/true_value
    print(f'The root is at x = : {c:.3f}, absolute error is {numerical_error}, relative error is {relative_error}')
    return c

code(0,100)

'''
the secant method is perfect for approximating zeroes of straight lines
however, a more correct approximation would be to use the 


'''