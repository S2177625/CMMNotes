# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:29:05 2023

@author: emily
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

Aj, omega_r,omega_i,t,yj, k = symbols("Aj omega_r omega_i t yj k") 
#given constants/variables
Aj_val = 0.1
yj_val = pi/8
omega_r_val = -0.62301963
omega_i_val = 24.03024141

# k = Initial force/ initial displacement
k_val = 100/Aj_val

x= Aj*exp(omega_r*t)*cos((omega_i*t)+yj)

F = -k*x
    
# F = -k*x bc of spring

# WORK DONE = INTEGRATION
#--------
# integrate between 0 and 10
#--------

Work_done= integrate(F,(t,0,10))
substitution = Work_done.subs({Aj:Aj_val, yj: yj_val, omega_r: omega_r_val, omega_i: omega_i_val,k:k_val})


print(substitution.evalf())