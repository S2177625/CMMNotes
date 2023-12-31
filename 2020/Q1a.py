import sympy as sym

#Defining symbols
y, dy, d2y, x, w, TA, y0 = sym.symbols('y dy 2dy x w TA y0')

#Using Equation 1B to get first and second derivatives
#The solution to the second derivative of Eq1B should be the same as the
#solution of Eq1A
y = (TA/w)*sym.cosh(x*(w/TA)) + y0 - TA/w
dy = sym.diff(y, x)
d2y_eq1B = sym.diff(dy, x)
#print(sym.solve(d2y_eq1B, x))

#This equation uses the first derivative from Eq1B, don't know if you can do that
d2y_eq1A = (w/TA)*sym.sqrt(1 + dy**2)
#print(sym.solve(d2y_eq1A, x))

#print(d2y_eq1B)
#print(d2y_eq1A)

#Checking if the solutions are the same
if sym.solve(d2y_eq1B, x) == sym.solve(d2y_eq1A, x):
    print('Equation 1B is the general solution')
else:
    print('Equation 1B is not the general solution')