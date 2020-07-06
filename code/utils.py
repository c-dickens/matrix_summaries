import numpy as np

def covariance_error(X_true,X_est,norm='fro'):
    """
    Evaluates the error in the returned covariance matrix under `norm' between the 
    true matrix and the estimated matrix.

    Input:
        true - ndarray of 'true' dataset as a matrix of size n x d
        est - ndarray of the sketched matrix of size m x d for some m << n.

    Output:
        error - float which evaluates 
        || X_true^T X_true - X_est^T X_est ||_2 /  || X_true^T X_true||_a
        where norm a = 'fro' (frobenius) or a=2 (spectral).
        This error test is taken from Section 5 of https://arxiv.org/pdf/1501.01711.pdf
        where "covariance error" corresponds to taking norm='fro' in this 
        function.
        Setting norm=2 just evaluates error on the leading direction.
    """
    true = X_true.T @ X_true
    est = X_est.T @ X_est
    numer = np.linalg.norm(true-est,ord=2)
    denom = np.linalg.norm(true,ord=norm)
    return numer/denom