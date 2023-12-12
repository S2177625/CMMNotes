#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:29:58 2023

@author: ben
"""

import numpy as np
import math



def sum(n):
    x=0
    pi_tru = math.pi
    for p in range(1,n):
        x = x + (1/(p*p))
        result = np.sqrt(6*x)
    print(result)
    true_err = ((pi_tru - result)/pi_tru)*100
    print(true_err)
    return
    

sum10 = sum(10)
sum100 = sum(100)
sum1000 = sum(1000)


