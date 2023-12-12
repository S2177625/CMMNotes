import numpy as np

def f(B):
    return np.cosh(B)*np.cos(B) + 1

#Bisection function
def bisection(a,b,N):
    # Check if a and b bound a root
    if f(a)*f(b) >= 0:
       print("a and b do not bound a root")
       return None 
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
           a_n = a_n
           b_n = m_n
        elif f(b_n)*f_m_n < 0:
           a_n = m_n
           b_n = b_n
        elif f_m_n == 0:
           print("Found exact solution.")
           return m_n
        else:
           print("Bisection method fails.")
           return None
    return (a_n + b_n)/2

B1 = bisection(0,3,10000)
B2 = bisection(4,5,10000)

E = 200E9
I = 3.255E-11
m = 7850
L = 0.9

f1 = np.sqrt(np.power(B1,4) * ((E*I)/(m*L**3))) / (2*np.pi)
f2 = np.sqrt(np.power(B2,4) * ((E*I)/(m*L**3))) / (2*np.pi)

print('B1:',B1,' f1:',f1)
print('B2:',B2,' f2:',f2)