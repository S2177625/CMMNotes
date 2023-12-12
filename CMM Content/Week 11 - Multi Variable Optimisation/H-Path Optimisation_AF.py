import sympy as sym
import numpy as np
from sympy import symbols, solve
import matplotlib.pyplot as plt

#E. McCarthy, A. Fragkou
def HOpt(F,dFx,dFy,x,y):


    hsym = symbols('hsym')

    xlist = []
    ylist = []
    flist = []
    dfxlist = []
    dfylist = []

    for i in range(0, 10, 1):
        xold = x
        yold = y

        dfx = dFx(x,y)
        dfy = dFy(x,y)

        #Create a function for the path to the top of the mountain.
        g = F(x+dfx*hsym, y+dfy*hsym)
        hexpr = sym.diff(g, hsym)

        hsolved = solve(hexpr)
        hopt = hsolved[0]

        x = xold + hopt*dfx
        y = yold + hopt*dfy

        Fxy = F(x, y)
        dfx = dFx(x,y)
        dfy = dFy(x,y)

        xlist.append(x)
        ylist.append(y)
        flist.append(Fxy)
        dfxlist.append(dfx)
        dfylist.append(dfy)
        #print(f"{dfx} <= {0.0001} and {dfy} <= {0.0001}")
        if dfx <= 0.0001 and dfy <= 0.0001:
            break
    return xlist[:], ylist[:], flist[:]



# CREATING FUNCTIONS
x, y = sym.symbols("x y")
F_sym = 2*x*y + 2*x - x**2 - 2*y**2

dFx_sym = F_sym.diff(x)
dFy_sym = F_sym.diff(y)
dFx = sym.lambdify((x, y), dFx_sym, 'numpy')
dFy = sym.lambdify((x, y), dFy_sym, 'numpy')

F = sym.lambdify((x, y), F_sym, 'numpy')

# INITIAL POINT
x_initial = 0.0
y_initial = 0.0

# OPTIMISATION
xlist, ylist, flist = HOpt(F, dFx, dFy, x_initial, y_initial)
maxima = [xlist[-1], ylist[-1]]
print(maxima)

# DISPLAY PURPOUSES ONLY
N = 33
SIZE = 3
AT_X, AT_Y = [2,1] 
xs = np.linspace(-SIZE+AT_X,SIZE+AT_X)   
ys = np.linspace(-SIZE+AT_Y,SIZE+AT_Y)    
X, Y = np.meshgrid(xs, ys)    

plt.scatter(xlist, ylist, color='red', marker='o')
plt.imshow(F(X,Y), cmap='viridis', extent=(xs[0], xs[-1], ys[0], ys[-1]), interpolation='nearest', origin='lower')

plt.show()