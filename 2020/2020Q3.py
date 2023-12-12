#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:05:26 2023

@author: ben
"""

import numpy as np
from sympy import symbols, solve

'''
n = number of years
P = present worth
A = anual payment
i = interest rate
'''
A_val = 25500
n_val = 6
P_val = 115000

P, i, n, A = symbols('P i n A')

def func(i):
    return P*((i*((1+i)**n))/(((1+i)**n))-1) - A

sol = solve(P*((i*((1+i)**n))/(((1+i)**n))-1) - A, i)

sol_value = sol[0].subs({A: A_val, P: P_val, n: n_val})

print(sol_value.evalf())




    