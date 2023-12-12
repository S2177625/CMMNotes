## Question 1 

import numpy as np
import matplotlib.pyplot as plt

# Set the number of terms for the series approximation
N = 100
# Initialize the sum of the series to zero
s = 0
# Create arrays to store computed values of pi, the number of terms, true error, and estimated error
pi_n = np.zeros(N)
nn = np.zeros(N)
error_true = np.zeros(N)
error_ext = np.zeros(N)

# Iterate over the range of 1 to N (inclusive) to compute the series
for i in range(1,N+1):
    # Store the previous approximation of pi
    pi_old = (s*6.0)**0.5
    # Update the series sum with the next term
    s = s + 1.0/i**2.0
    # Compute the new approximation of pi using the updated sum
    pi_n[i-1] = (s*6.0)**0.5
    # Store the number of terms used so far
    nn[i-1] = i 
    # Compute the true error using the absolute difference between the known pi and our approximation
    error_true[i-1] = np.absolute(pi_n[i-1] - np.pi)
    # Compute the estimated error using the absolute difference between the current and previous approximation of pi
    error_ext[i-1]  = np.absolute(pi_n[i-1] - pi_old) 
    # Print the number of terms, the current approximation of pi, the true error, and the estimated error
    print(i,pi_n[i-1],error_true[i-1],error_ext[i-1])

# Plot the approximation of pi over the number of terms
plt.figure()
plt.plot(nn,pi_n)
plt.xlabel('number of iterations')
plt.ylabel('Value')
# Plot both true and estimated errors using a logarithmic scale
plt.figure()
plt.loglog(nn,error_true,'-b',nn,error_ext,'.r')
plt.xlabel('Number of itertations')
plt.ylabel('error in approximation')
# Display the plots
plt.show()


''''This code approximates π using a series and analyzes convergence by plotting true and 
estimated errors. It iteratively adds terms (1/i²) to the series, updating π's approximation
 and calculating true and estimated errors based on the difference from numpy's π and the 
 previous approximation, respectively. Plots show π's value and error convergence over 
 iterations.
'''