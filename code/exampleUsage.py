import sys
from numpy.linalg import norm
from numpy import dot, zeros

from syntheticDataMaker import SyntheticDataMaker
from frequentDirections import FrequentDirections

n = 500
d = 100
ell = 20
k = 5

# Generating the full matrix for error performance.
A = zeros((n,d))

# this is only needed for generating input vectors
dataMaker = SyntheticDataMaker()
dataMaker.initBeforeMake(d, k, signal_to_noise_ratio=10.0)                                                                                                                                                                                                                                                                                                                         

# This is where the sketching actually happens
sketcher = FrequentDirections(d,ell)
for i in range(n):
    row = dataMaker.makeRow()
    sketcher.append(row)
    A[i,:] = row
sketch = sketcher.get()

# Here is where you do something with the sketch.
# The sketch is an ell by d matrix 
# For example, you can compute an approximate covariance of the input 
# matrix like this:

approxCovarianceMatrix = dot(sketch.transpose(),sketch)
print(approxCovarianceMatrix)
print('**********     ERRORS     **********')
true_cov = A.T@A
error_norm = norm(true_cov - approxCovarianceMatrix,ord=2)
true_norm =  norm(true_cov,ord=2)
print('Spectral error: ', error_norm/true_norm)
 





