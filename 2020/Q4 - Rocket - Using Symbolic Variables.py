
import sympy as sym

u = 1.8 * 10**3
m_0 = 160 * 10**3
q = 2.5 * 10**3
g = 9.81

t = sym.symbols("t")
Ft_sym = u * sym.log((m_0)/(m_0-q*t)) -g*t

#dFy_sym = F_sym.diff(y)
#dFy = sym.lambdify((x, y), dFy_sym, 'numpy')
#F = sym.lambdify((x, y), F_sym, 'numpy')

print(sym.integrate(Ft_sym, (t, 0, 30)))