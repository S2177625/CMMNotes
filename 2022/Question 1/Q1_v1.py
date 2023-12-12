import math
from math import pi
import numpy as np

def equation(N):
    sum = 0
    sum1 = 0
    for i in range(1, N+1):
        sum = (1/(i**2))
        if sum1 != 0:
            error = ((sum1-sum)/sum1) * 100
        sum1 = sum + sum
    pi_approx = math.sqrt(6*sum)
    true_error = ((pi - pi_approx)/pi)*100
    return pi_approx, "true error = " + str(true_error) + "%", error

print(equation(1000))