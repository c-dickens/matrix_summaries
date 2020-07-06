import numpy as np
from utils import covariance_error
from sparse_coresets import SparseCoreset

A = np.random.randn(1000,10)
sketcher = SparseCoreset(A,0.5,0.1,sample_size=50)
sketcher.stream(block_size=250)
B = sketcher.coreset
err_fro = covariance_error(A, B,norm='fro')
err_spc = covariance_error(A, B,norm=2)
print('Frobenius Error: ', err_fro)
print('Spectral Error: ', err_spc)


