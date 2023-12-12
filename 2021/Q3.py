MAX_ITER = 100000

def func( x ):
    return (x**2 + 5*x - 4)

def Code( a , b):
    if func(a) * func(b) >= 0:
        print("You have not assumed correct values of a and b")
        return 0
    c = a
    for i in range(MAX_ITER):
        c = (a * func(b) - b * func(a))/ (func(b) - func(a))
        if func(b) == 0:
            break
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c
    print("The value of root is : " , '%.4f' %c)
    
Code(0,2)