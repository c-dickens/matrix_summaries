import numpy as np
from math import sqrt

class MatrixSampler:
    """
    Performs matrix sampling in either leverage or uniform formats.
    """
    def __init__(self,mat,sample_size):
        self.A = mat
        self.n,self.d = self.A.shape
        self.sample_size = sample_size
    
    def uniform_sampling(self):
        """
        Performs uniform sampling on self.A
        """
        sampled_ids = np.random.choice(self.n, size=self.sample_size)
        return sqrt(self.n / self.sample_size)*self.A[sampled_ids,:]
    
    def leverage_score_sampling(self):
        """
        Performs leverage score sampling on self.A
        Leverage scores are evaluated by the row norms of an orthonormal
        basis which are then used for weighted sampling.
        """
        U,_,_ = np.linalg.svd(self.A,full_matrices=False)
        scores = np.linalg.norm(U, axis=1)**2
        sampled_ids = np.random.choice(self.n,
                                    size=self.sample_size,
                                    p=scores/scores.sum())
        sketch = self.A[sampled_ids,:] 
        sketch /= np.sqrt(self.sample_size*scores[sampled_ids])[:,None]
        return sketch

