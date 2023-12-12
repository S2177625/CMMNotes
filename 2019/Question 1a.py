# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:17:13 2023

@author: emily
"""


#-----------------------------
# QUESTION 1 A
#-----------------------------
import numpy as np

c_m = 12
s_m = 1500

coeff = (1, 2*c_m, 3* s_m, c_m*s_m, s_m**2)
roots = np.roots(coeff)
omega_r = np.real(roots)
omega_i = np.imag(roots)


print(omega_r,omega_i)