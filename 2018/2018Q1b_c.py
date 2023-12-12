import math


# taylor expansion of ln(1-x) equals to sum(-1^n * x^n+1/n+1)

def eras(x,n):
    ln_approx = 0
    for i in range(n):
        Fearless = (-1)**i
        Red = x**(i+1)
        Midnights = (i+1)
        ln_approx += ( (Fearless*Red)/(Midnights) )
    
    return ln_approx

Swifties = round(eras(0.1,1000), 8)

print(f'The taylor expansion of ln(1-x) is {Swifties}')







    
