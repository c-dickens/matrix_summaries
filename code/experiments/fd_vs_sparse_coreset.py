import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tabulate import tabulate
from syntheticDataMaker import SyntheticDataMaker
from frequentDirections import FrequentDirections
from utils import covariance_error
from sparse_coresets import SparseCoreset
from matrix_sampler import MatrixSampler

def main():
    # Generate some input data
    n = 1000
    d = 32
    k = 28
    print('*'*20)

    dataMaker = SyntheticDataMaker()
    dataMaker.initBeforeMake(d, k, signal_to_noise_ratio=10.0)      
    A = dataMaker.makeMatrix(n)
    

    # Fit the FD sketch.
    ell = d
    fd_sketcher = FrequentDirections(d,ell)
    for i in range(n):
        row = A[i,:]
        fd_sketcher.append(row)
    fd_sketch = fd_sketcher.get()
    fd_cov_error = covariance_error(A,fd_sketch)

    # # Fit the sparse coreset sketcher.
    # sparse_sketcher = SparseCoreset(A,1.0,2.0,sample_size=ell)
    # sparse_sketcher.stream(block_size=n)
    # sparse_sketch = sparse_sketcher.coreset
    # sparse_cov_error = covariance_error(A,sparse_sketch)
    # print('Coreset size: ',  sparse_sketch.shape)
    # print('SparseCoreset covariance error: ', sparse_cov_error)

    # Gaussian Sketch
    S = np.random.randn(ell,n) / np.sqrt(ell)
    SA = S@A
    gauss_cov_error = covariance_error(A,SA)

    # Uniform sampling
    sampler = MatrixSampler(A,ell)
    S_uniform = sampler.uniform_sampling()
    S_leverage = sampler.leverage_score_sampling()
    uniform_cov_error = covariance_error(A, S_uniform)
    leverage_error = covariance_error(A, S_leverage)
    # print('Uniform cov error: ', uniform_cov_error)
    # print('Leverage cov error: ', leverage_error)

    table = [
        ['FreqDirs', fd_sketch.shape, fd_cov_error],
        ['Uniform', S_uniform.shape, uniform_cov_error],
        ['Leverage', S_leverage.shape, leverage_error],
        ['Gaussian', SA.shape, gauss_cov_error]
    ]

    print('*'*10, '     RESULTS     ', '*'*10)
    h = ['Sketch', 'Size', 'Error']
    print(tabulate(table, headers=h))




    
    

if __name__ == '__main__':
    main()